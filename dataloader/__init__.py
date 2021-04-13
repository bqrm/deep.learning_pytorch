# -*- coding: utf-8 -*-
# Version:		python 3.6.5
# Description:	
# Author:		bqrmtao@qq.com
# date:			2019/12/13

from torch.utils.data import DataLoader





def make_data_loader(args, **kwargs):
    if "pascal" == args.dataset:
        from dataloader.dataset.pascal_voc import PascalVocSeg

        train_set = PascalVocSeg(root_dir=args.root_dir, crop_size=args.crop_size, split="train")
        valid_set = PascalVocSeg(root_dir=args.root_dir, crop_size=args.crop_size, split="val")

        if args.use_sbd:
            raise NotImplementedError

        num_class = train_set.num_class
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        valid_loader = None
    elif "image_net" == args.dataset:
        from dataloader.dataset.mini_image_net import mini_image_net

        train_set = MiniImageNet(root_dir=args.root_dir, crop_size=args.crop_size, split="train")
        test_set = MiniImageNet(root_dir=args.root_dir, crop_size=args.crop_size, split="test")
        valid_set = MiniImageNet(root_dir=args.root_dir, crop_size=args.crop_size, split="val")

        num_class = train_set.num_class
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    else:
        raise NotImplementedError

    return train_loader, test_loader, valid_loader, num_class
















