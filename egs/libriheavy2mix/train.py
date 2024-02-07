# Copyright (c) 2021 Brno University of Technology
# Copyright (c) 2021 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved
# By Katerina Zmolikova, August 2021.

# This code is based on Asteroid train.py,
# which is released under the following MIT license:
# https://github.com/asteroid-team/asteroid/blob/master/LICENSE

import argparse
import json
import os

import pytorch_lightning as pl
import torch
from asteroid.engine.optimizers import make_optimizer
from asteroid.losses import singlesrc_neg_sisdr, singlesrc_neg_snr
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from datasets.libriheavymix_informed import LibriheavyMixInformed
from models.system import SystemInformed
from models.td_speakerbeam import TimeDomainSpeakerBeam

# Keys which are not in the conf.yml file can be added here.
# In the hierarchical dictionary created when parsing, the key `key` can be
# found at dic['main_args'][key]

# By default train.py will use all available GPUs. The `id` option in run.sh
# will limit the number of available GPUs for train.py .
parser = argparse.ArgumentParser()
parser.add_argument(
    "--exp_dir", default="exp/tmp", help="Full path to save best validation model"
)


def neg_sisdr_loss_wrapper(est_targets, targets):
    return singlesrc_neg_sisdr(est_targets[:, 0], targets[:, 0]).mean()
    # return singlesrc_neg_sisdr(est_targets[:, 0], targets[:, 0]).mean()


def main(conf):
    train_set = LibriheavyMixInformed(
        mixscp=conf["train"]["train_mixscp"],
        mix2spk=conf["train"]["train_mix2spk"],
        spk2src=conf["train"]["train_spk2src"],
        spk2spk=conf["train"]["train_spk2spk"],
        enrollments=None,
        sample_rate=conf["train"]["train_sample_rate"],
        segment=conf["train"]["train_segment"],
        segment_aux=conf["train"]["train_segment_aux"],
        train=True,
    )

    val_set = LibriheavyMixInformed(
        mixscp=conf["dev"]["dev_mixscp"],
        mix2spk=conf["dev"]["dev_mix2spk"],
        spk2src=conf["dev"]["dev_spk2src"],
        spk2spk=conf["dev"]["dev_spk2spk"],
        enrollments=conf["dev"]["dev_enrollments"],
        sample_rate=conf["dev"]["dev_sample_rate"],
        train=False,
        test=False,
    )

    train_loader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=conf["training"]["batch_size"],
        num_workers=conf["training"]["num_workers"],
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        shuffle=False,
        batch_size=1,  # conf["training"]["batch_size"],
        num_workers=conf["training"]["num_workers"],
        drop_last=True,
    )
    conf["masknet"].update({"n_src": 2})

    model = TimeDomainSpeakerBeam(
        **conf["filterbank"],
        **conf["masknet"],
        sample_rate=conf["train"]["train_sample_rate"],
        **conf["enroll"]
    )
    optimizer = make_optimizer(model.parameters(), **conf["optim"])
    # Define scheduler
    scheduler = None
    if conf["training"]["half_lr"]:
        scheduler = ReduceLROnPlateau(
            optimizer=optimizer,
            factor=0.5,
            patience=conf["training"]["reduce_patience"],
        )
    # Just after instantiating, save the args. Easy loading in the future.
    exp_dir = conf["main_args"]["exp_dir"]
    os.makedirs(exp_dir, exist_ok=True)
    conf_path = os.path.join(exp_dir, "conf.yml")
    with open(conf_path, "w") as outfile:
        yaml.safe_dump(conf, outfile)

    # Define Loss function.
    loss_func = neg_sisdr_loss_wrapper
    system = SystemInformed(
        model=model,
        loss_func=loss_func,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        scheduler=scheduler,
        config=conf,
    )

    # Define callbacks
    callbacks = []
    checkpoint_dir = os.path.join(exp_dir, "checkpoints/")
    checkpoint = ModelCheckpoint(
        checkpoint_dir, monitor="val_loss", mode="min", save_top_k=-1, verbose=True
    )
    callbacks.append(checkpoint)
    if conf["training"]["early_stop"]:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=conf["training"]["stop_patience"],
                verbose=True,
            )
        )
    callbacks.append(LearningRateMonitor())

    # Don't ask GPU if they are not available.
    gpus = -1 if torch.cuda.is_available() else None

    trainer = pl.Trainer(
        max_epochs=conf["training"]["epochs"],
        callbacks=callbacks,
        default_root_dir=exp_dir,
        devices=gpus,
        limit_train_batches=1.0,  # Useful for fast experiment
        gradient_clip_val=5.0,
        strategy="ddp_find_unused_parameters_true",
    )
    trainer.fit(system)

    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(exp_dir, "best_k_models.json"), "w") as f:
        json.dump(best_k, f, indent=0)

    state_dict = torch.load(checkpoint.best_model_path)
    system.load_state_dict(state_dict=state_dict["state_dict"])
    system.cpu()

    to_save = system.model.serialize()
    to_save.update(train_set.get_infos())
    torch.save(to_save, os.path.join(exp_dir, "best_model.pth"))


if __name__ == "__main__":
    from pprint import pprint

    import yaml
    from asteroid.utils import parse_args_as_dict, prepare_parser_from_dict

    # We start with opening the config file conf.yml as a dictionary from
    # which we can create parsers. Each top level key in the dictionary defined
    # by the YAML file creates a group in the parser.
    with open("local/conf.yml") as f:
        def_conf = yaml.safe_load(f)
    parser = prepare_parser_from_dict(def_conf, parser=parser)
    # Arguments are then parsed into a hierarchical dictionary (instead of
    # flat, as returned by argparse) to facilitate calls to the different
    # asteroid methods (see in main).
    # plain_args is the direct output of parser.parse_args() and contains all
    # the attributes in an non-hierarchical structure. It can be useful to also
    # have it so we included it here but it is not used.
    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)
    pprint(arg_dic)
    main(arg_dic)
