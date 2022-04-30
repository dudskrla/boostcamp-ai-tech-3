from .transforms import *
import torch


class BaseTrashTTA:
    """ author: @shonenkov """
    image_size = 1024

    def augment(self, image):
        raise NotImplementedError
    
    def batch_augment(self, images):
        raise NotImplementedError
    
    def deaugment_boxes(self, boxes):
        raise NotImplementedError

class TTAHorizontalFlip(BaseTrashTTA):
    """ author: @shonenkov """

    def augment(self, image):
        return image.flip(1)
    
    def batch_augment(self, images):
        return images.flip(2)
    
    def deaugment_boxes(self, boxes):
        boxes[:, [1,3]] = self.image_size - boxes[:, [3,1]]
        return boxes

class TTAVerticalFlip(BaseTrashTTA):
    """ author: @shonenkov """
    
    def augment(self, image):
        return image.flip(2)
    
    def batch_augment(self, images):
        return images.flip(3)
    
    def deaugment_boxes(self, boxes):
        boxes[:, [0,2]] = self.image_size - boxes[:, [2,0]]
        return boxes
    
class TTARotate90(BaseTrashTTA):
    """ author: @shonenkov """
    
    def augment(self, image):
        return torch.rot90(image, 1, (1, 2))

    def batch_augment(self, images):
        return torch.rot90(images, 1, (2, 3))
    
    def deaugment_boxes(self, boxes):
        res_boxes = boxes.copy()
        res_boxes[:, [0,2]] = self.image_size - boxes[:, [1,3]]
        res_boxes[:, [1,3]] = boxes[:, [2,0]]
        return res_boxes

class TTAResize(BaseTrashTTA):
    def __init__(self, img_size):
        self.img_size = img_size
        self.resize = torchvision.transforms.Resize(img_size)

    def augment(self, image):
        img = self.resize(image)
        return img 

    def batch_augment(self, image):
        img = self.resize(image)
        return img 

    def clip_boxes_(boxes, img_size):
        height, width = img_size
        clip_upper = np.array([height, width] * 2, dtype=boxes.dtype)
        np.clip(boxes, 0, clip_upper, out=boxes)

    def deaugment_boxes(self, boxes):
        width, height = 1024, 1024

        img_scale_y = self.img_size / height
        img_scale_x = self.img_size / width
        img_scale = min(img_scale_y, img_scale_x)
        scaled_h = int(height * img_scale)
        scaled_w = int(width * img_scale)

        bbox = boxes.copy()
        bbox[:, :4] *= img_scale
        bbox_bound = (min(scaled_h, self.img_size), min(scaled_w, self.img_size))
        clip_boxes_(bbox, bbox_bound)  # crop to bounds of target image or letter-box, whichever is smaller
        # valid_indices = (bbox[:, :2] < bbox[:, 2:4]).all(axis=1)
        # bbox = bbox[valid_indices, :]

        return bbox

class TTACompose(BaseTrashTTA):
    """ author: @shonenkov """
    def __init__(self, transforms):
        self.transforms = transforms
        
    def augment(self, image):
        for transform in self.transforms:
            image = transform.augment(image)
        return image
    
    def batch_augment(self, images):
        for transform in self.transforms:
            images = transform.batch_augment(images)
        return images
    
    def prepare_boxes(self, boxes):
        result_boxes = boxes.copy()
        result_boxes[:,0] = np.min(boxes[:, [0,2]], axis=1)
        result_boxes[:,2] = np.max(boxes[:, [0,2]], axis=1)
        result_boxes[:,1] = np.min(boxes[:, [1,3]], axis=1)
        result_boxes[:,3] = np.max(boxes[:, [1,3]], axis=1)
        return result_boxes
    
    def deaugment_boxes(self, boxes):
        for transform in self.transforms[::-1]:
            boxes = transform.deaugment_boxes(boxes)
        return self.prepare_boxes(boxes)



from itertools import product

tta_transforms = []

for tta_combination in product(# [# TTAResize(img_size=512), 
                            #    TTAResize(img_size=768), 
                            #    TTAResize(img_size=1024),
                            #    TTAResize(img_size=1280), 
                            #    TTAResize(img_size=1536)
                            #    ],
                               [TTAHorizontalFlip(), None], 
                               [TTAVerticalFlip(), None],
                               [TTARotate90(), None],
                               ):
    tta_transforms.append(TTACompose([tta_transform for tta_transform in tta_combination if tta_transform]))