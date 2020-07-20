import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

def get_train_transforms():
    return A.Compose(
        [A.ShiftScaleRotate(scale_limit=(-0.5, 0.5), rotate_limit=0, shift_limit=0., p=0.5, border_mode=0),
            A.RandomRotate90(p=0.5),
            A.Resize(1024, 1024, p=1),
            A.VerticalFlip(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.OneOf([
                    A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit= 0.2,
                                         val_shift_limit=0.2, p=0.9),
                    A.RandomBrightnessContrast(brightness_limit=0.2,
                                               contrast_limit=0.2, p=0.9),
                    A.RandomGamma(p=0.9),
            ],p=0.25),
            A.OneOf([
                A.IAASharpen(alpha=(0.1, 0.3), p=0.5),
                A.CLAHE(p=0.8),
                #A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                #A.GaussianBlur(blur_limit=3, p=0.5),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.3),
            ], p=0.0),
         A.Cutout(num_holes=8, max_h_size=64, max_w_size=64, fill_value=0, p=0.5),
         ToTensorV2(p=1.0),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )
    )


def get_valid_transforms():
    return A.Compose(
        [
            A.Resize(height=1024, width=1024, p=1.0),
            ToTensorV2(p=1.0),
        ],
        p=1.0,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            min_area=0,
            min_visibility=0,
            label_fields=['labels']
        )
    )
