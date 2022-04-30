#!/usr/bin/env python
""" COCO validation script

Hacked together by Ross Wightman (https://github.com/rwightman)
"""
import argparse
import time
import torch
import torch.nn.parallel
from contextlib import suppress
import pandas as pd 

from effdet import create_model, create_evaluator, create_dataset, create_loader
from effdet.data import resolve_input_config
from timm.utils import AverageMeter, setup_default_logging
from timm.models.layers import set_layer_config

has_apex = False
try:
    from apex import amp
    has_apex = True
except ImportError:
    pass

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

torch.backends.cudnn.benchmark = True


def add_bool_arg(parser, name, default=False, help=''):  # FIXME move to utils
    dest_name = name.replace('-', '_')
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--' + name, dest=dest_name, action='store_true', help=help)
    group.add_argument('--no-' + name, dest=dest_name, action='store_false', help=help)
    parser.set_defaults(**{dest_name: default})


parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument('root', metavar='DIR',
                    help='path to dataset root')
parser.add_argument('--dataset', default='coco', type=str, metavar='DATASET',
                    help='Name of dataset (default: "coco"')
parser.add_argument('--split', default='val',
                    help='validation split')
parser.add_argument('--model', '-m', metavar='MODEL', default='tf_efficientdet_d1',
                    help='model architecture (default: tf_efficientdet_d1)')
add_bool_arg(parser, 'redundant-bias', default=None,
                    help='override model config for redundant bias layers')
add_bool_arg(parser, 'soft-nms', default=None, help='override model config for soft-nms')
parser.add_argument('--num-classes', type=int, default=None, metavar='N',
                    help='Override num_classes in model config if set. For fine-tuning from pretrained.')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--img-size', default=None, type=int,
                    metavar='N', help='Input image dimension, uses model default if empty')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float,  nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='bilinear', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--fill-color', default=None, type=str, metavar='NAME',
                    help='Image augmentation fill (background) color ("mean" or int)')
parser.add_argument('--log-freq', default=10, type=int,
                    metavar='N', help='batch logging frequency (default: 10)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--use-ema', dest='use_ema', action='store_true',
                    help='use ema version of weights if present')
parser.add_argument('--amp', action='store_true', default=False,
                    help='Use AMP mixed precision. Defaults to Apex, fallback to native Torch AMP.')
parser.add_argument('--apex-amp', action='store_true', default=False,
                    help='Use NVIDIA Apex AMP mixed precision')
parser.add_argument('--native-amp', action='store_true', default=False,
                    help='Use Native Torch AMP mixed precision')
parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                    help='convert model torchscript for inference')
parser.add_argument('--results', default='', type=str, metavar='FILENAME',
                    help='JSON filename for evaluation results')

# TTA start 
import ensemble_boxes
from effdet.data.TTA import *

def make_tta_predictions(input, bench, target, score_threshold=0.25):
    with torch.no_grad():
        images = input.cuda().float()
        # images = torch.stack(input).float().cuda()
        predictions = []
        for tta_transform in tta_transforms:
            result = []

            # img = img.clone().squeeze()
            # images = to_pil_image(img)
            # img_info = {'img_scale': torch.tensor([1]*images.shape[0]).float().cuda(),
            #             'img_size' : torch.tensor([1]*images.shape[0]).float().cuda()}


            tta_transformed = tta_transform.batch_augment(images.clone())
            img_info = {'img_size': torch.tensor([[1024, 1024]]).float().cuda(),
                        'img_scale' : torch.tensor([1.]).float().cuda()}


            det = bench(tta_transformed, img_info=target) # images.clone()), img_info=target) # img_info=torch.tensor([1]*images.shape[0]).float().cuda())

            for i in range(images.shape[0]):
                boxes = det[i].detach().cpu().numpy()[:,:4]    
                scores = det[i].detach().cpu().numpy()[:,4]
                indexes = np.where(scores > score_threshold)[0]

                boxes = boxes[indexes]
                boxes[:, 2] = boxes[:, 2] + boxes[:, 0]
                boxes[:, 3] = boxes[:, 3] + boxes[:, 1]

                boxes = tta_transform.deaugment_boxes(boxes.copy())

                result.append({
                    'boxes': boxes,
                    'scores': scores[indexes],
                })
            predictions.append(result)
    return predictions


def run_wbf(predictions, image_index, image_size=1024, iou_thr=0.44, skip_box_thr=0.43, weights=None):
    boxes = [(prediction[image_index]['boxes']/(image_size-1)).tolist() for prediction in predictions]
    scores = [prediction[image_index]['scores'].tolist() for prediction in predictions]
    labels = [np.ones(prediction[image_index]['scores'].shape[0]).astype(int).tolist() for prediction in predictions]
    boxes, scores, labels = ensemble_boxes.ensemble_boxes_wbf.weighted_boxes_fusion(boxes, scores, labels, weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    boxes = boxes*(image_size-1)
    return boxes, scores, labels

def format_prediction_string(boxes, scores):
    pred_strings = []
    for j in zip(scores, boxes):
        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))
    return " ".join(pred_strings)
# TTA end


def validate(args):
    setup_default_logging()

    if args.amp:
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True
    assert not args.apex_amp or not args.native_amp, "Only one AMP mode should be set."
    args.pretrained = args.pretrained or not args.checkpoint  # might as well try to validate something
    args.prefetcher = not args.no_prefetcher

    # create model
    with set_layer_config(scriptable=args.torchscript):
        extra_args = {}
        if args.img_size is not None:
            extra_args = dict(image_size=(args.img_size, args.img_size))
        bench = create_model(
            args.model,
            bench_task='predict',
            num_classes=args.num_classes,
            pretrained=args.pretrained,
            redundant_bias=args.redundant_bias,
            soft_nms=args.soft_nms,
            checkpoint_path=args.checkpoint,
            checkpoint_ema=args.use_ema,
            **extra_args,
        )
    model_config = bench.config

    param_count = sum([m.numel() for m in bench.parameters()])
    print('Model %s created, param count: %d' % (args.model, param_count))

    bench = bench.cuda()

    amp_autocast = suppress
    if args.apex_amp:
        bench = amp.initialize(bench, opt_level='O1')
        print('Using NVIDIA APEX AMP. Validating in mixed precision.')
    elif args.native_amp:
        amp_autocast = torch.cuda.amp.autocast
        print('Using native Torch AMP. Validating in mixed precision.')
    else:
        print('AMP not enabled. Validating in float32.')

    if args.num_gpu > 1:
        bench = torch.nn.DataParallel(bench, device_ids=list(range(args.num_gpu)))

    dataset = create_dataset(args.dataset, args.root, args.split)
    input_config = resolve_input_config(args, model_config)
    loader = create_loader(
        dataset,
        input_size=input_config['input_size'],
        batch_size=args.batch_size,
        use_prefetcher=args.prefetcher,
        interpolation=input_config['interpolation'],
        fill_color=input_config['fill_color'],
        mean=input_config['mean'],
        std=input_config['std'],
        num_workers=args.workers,
        pin_mem=args.pin_mem)

    evaluator = create_evaluator(args.dataset, dataset, pred_yxyx=False)
    bench.eval()
    batch_time = AverageMeter()
    end = time.time()
    last_idx = len(loader) - 1

    results = []
    with torch.no_grad():
        for i, (input, target) in enumerate(loader):
            with amp_autocast():
                output = bench(input, img_info=target)

                ## tta code start 
                predictions = make_tta_predictions(input, bench, target)

                for i, image in enumerate(input):
                    boxes, scores, labels = run_wbf(predictions, image_index=i)
                    boxes = (boxes*2).round().astype(np.int32).clip(min=0, max=1023)
                    image_id = str(i) # target['image_idx']

                    boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
                    boxes[:, 3] = boxes[:, 3] - boxes[:, 1]

                    result = {
                        'image_id': image_id,
                        'PredictionString': format_prediction_string(boxes, scores)
                    }
                    results.append(result)
                ## tta code end 


            evaluator.add_predictions(output, target)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.log_freq == 0 or i == last_idx:
                print(
                    'Test: [{0:>4d}/{1}]  '
                    'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    .format(
                        i, len(loader), batch_time=batch_time,
                        rate_avg=input.size(0) / batch_time.avg)
                )

    mean_ap = 0.
    if dataset.parser.has_labels:
        mean_ap = evaluator.evaluate(output_result_file=args.results)
    else:
        evaluator.save(args.results)
    
    ## tta code start
    test_df = pd.DataFrame(results, columns=['image_id', 'PredictionString'])
    test_df.to_csv('submission.csv', index=False)
    ## tta code end 

    return mean_ap


def main():
    args = parser.parse_args()
    validate(args)


if __name__ == '__main__':
    main()

