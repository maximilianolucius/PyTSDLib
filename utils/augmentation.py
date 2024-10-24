import numpy as np
from tqdm import tqdm
from scipy.interpolate import CubicSpline


def jitter(x, sigma=0.03):
    """
    Adds random Gaussian noise to the input data.
    """
    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)


def scaling(x, sigma=0.1):
    """
    Applies scaling to the input data by multiplying each feature with a random factor.
    """
    factor = np.random.normal(loc=1., scale=sigma, size=(x.shape[0], x.shape[2]))
    return x * factor[:, np.newaxis, :]


def rotation(x):
    """
    Rotates the features of the input data by randomly flipping and shuffling dimensions.
    """
    flip = np.random.choice([-1, 1], size=(x.shape[0], x.shape[2]))
    rotate_axis = np.arange(x.shape[2])
    np.random.shuffle(rotate_axis)
    return flip[:, np.newaxis, :] * x[:, :, rotate_axis]


def permutation(x, max_segments=5, seg_mode="equal"):
    """
    Randomly permutes segments of the input data to introduce variability.
    """
    orig_steps = np.arange(x.shape[1])
    num_segs = np.random.randint(1, max_segments, size=x.shape[0])
    ret = np.zeros_like(x)

    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[1] - 2, num_segs[i] - 1, replace=False)
                splits = np.split(orig_steps, np.sort(split_points))
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[warp]
        else:
            ret[i] = pat
    return ret


def magnitude_warp(x, sigma=0.2, knot=4):
    """
    Warps the magnitude of the input data using cubic splines.
    """
    orig_steps = np.arange(x.shape[1])
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot + 2, x.shape[2]))
    warp_steps = np.linspace(0, x.shape[1] - 1, num=knot + 2).reshape(-1, 1)

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        warper = np.array(
            [CubicSpline(warp_steps.ravel(), random_warps[i, :, dim])(orig_steps) for dim in range(x.shape[2])]).T
        ret[i] = pat * warper
    return ret


def time_warp(x, sigma=0.2, knot=4):
    """
    Warps the time dimension of the input data using cubic splines.
    """
    orig_steps = np.arange(x.shape[1])
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot + 2, x.shape[2]))
    warp_steps = np.linspace(0, x.shape[1] - 1, num=knot + 2).reshape(-1, 1)

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            time_warp = CubicSpline(warp_steps.ravel(), warp_steps.ravel() * random_warps[i, :, dim])(orig_steps)
            scale = (x.shape[1] - 1) / time_warp[-1]
            ret[i, :, dim] = np.interp(orig_steps, np.clip(scale * time_warp, 0, x.shape[1] - 1), pat[:, dim])
    return ret


def window_slice(x, reduce_ratio=0.9):
    """
    Randomly slices a window of the input data and resamples it to the original size.
    """
    target_len = int(np.ceil(reduce_ratio * x.shape[1]))
    if target_len >= x.shape[1]:
        return x

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        start = np.random.randint(0, x.shape[1] - target_len)
        for dim in range(x.shape[2]):
            ret[i, :, dim] = np.interp(np.linspace(0, target_len, num=x.shape[1]), np.arange(target_len),
                                       pat[start:start + target_len, dim])
    return ret


def window_warp(x, window_ratio=0.1, scales=[0.5, 2.0]):
    """
    Warps a random window of the input data by applying scaling to the window.
    """
    warp_scales = np.random.choice(scales, size=x.shape[0])
    warp_size = np.ceil(window_ratio * x.shape[1]).astype(int)

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        start = np.random.randint(1, x.shape[1] - warp_size - 1)
        end = start + warp_size
        for dim in range(x.shape[2]):
            window_seg = np.interp(np.linspace(0, warp_size - 1, int(warp_size * warp_scales[i])), np.arange(warp_size),
                                   pat[start:end, dim])
            warped = np.concatenate([pat[:start, dim], window_seg, pat[end:, dim]])
            ret[i, :, dim] = np.interp(np.arange(x.shape[1]), np.linspace(0, x.shape[1] - 1, warped.size), warped)
    return ret


def spawner(x, labels, sigma=0.05):
    """
    Augments the data using the SPAWNER algorithm, which selects patterns from the same class and combines paths.
    """
    import utils.dtw as dtw
    ret = np.zeros_like(x)
    random_points = np.random.randint(1, x.shape[1] - 1, size=x.shape[0])

    for i, pat in enumerate(x):
        choices = np.where(labels == labels[i])[0]
        if choices.size > 1:
            random_sample = x[np.random.choice(choices)]
            path1 = dtw.dtw(pat[:random_points[i]], random_sample[:random_points[i]], dtw.RETURN_PATH)
            path2 = dtw.dtw(pat[random_points[i]:], random_sample[random_points[i]:], dtw.RETURN_PATH)
            combined = np.concatenate([np.vstack(path1), np.vstack(path2 + random_points[i])], axis=1)
            mean_pattern = np.mean([pat[combined[0]], random_sample[combined[1]]], axis=0)
            for dim in range(x.shape[2]):
                ret[i, :, dim] = np.interp(np.arange(x.shape[1]), np.linspace(0, x.shape[1] - 1, mean_pattern.shape[0]),
                                           mean_pattern[:, dim])
        else:
            ret[i] = pat
    return jitter(ret, sigma=sigma)


def run_augmentation(x, y, args):
    """
    Runs data augmentation multiple times based on the augmentation ratio.
    """
    np.random.seed(args.seed)
    x_aug, y_aug = x, y

    if args.augmentation_ratio > 0:
        for _ in range(args.augmentation_ratio):
            x_temp, tags = augment(x, y, args)
            x_aug = np.append(x_aug, x_temp, axis=0)
            y_aug = np.append(y_aug, y, axis=0)
            print(f"Augmentation round done with tags: {tags}")
    return x_aug, y_aug


def augment(x, y, args):
    """
    Applies various augmentation techniques based on the arguments provided.
    """
    augmentation_tags = ""
    if args.jitter:
        x = jitter(x)
        augmentation_tags += "_jitter"
    if args.scaling:
        x = scaling(x)
        augmentation_tags += "_scaling"
    if args.rotation:
        x = rotation(x)
        augmentation_tags += "_rotation"
    if args.permutation:
        x = permutation(x)
        augmentation_tags += "_permutation"
    if args.magwarp:
        x = magnitude_warp(x)
        augmentation_tags += "_magwarp"
    if args.timewarp:
        x = time_warp(x)
        augmentation_tags += "_timewarp"
    if args.windowslice:
        x = window_slice(x)
        augmentation_tags += "_windowslice"
    if args.windowwarp:
        x = window_warp(x)
        augmentation_tags += "_windowwarp"
    if args.spawner:
        x = spawner(x, y)
        augmentation_tags += "_spawner"

    return x, augmentation_tags


def run_augmentation_single(x, y, args):
    # print("Augmenting %s"%args.data)
    np.random.seed(args.seed)

    x_aug = x
    y_aug = y


    if len(x.shape)<3:
        # Augmenting on the entire series: using the input data as "One Big Batch"
        #   Before  -   (sequence_length, num_channels)
        #   After   -   (1, sequence_length, num_channels)
        # Note: the 'sequence_length' here is actually the length of the entire series
        x_input = x[np.newaxis,:]
    elif len(x.shape)==3:
        # Augmenting on the batch series: keep current dimension (batch_size, sequence_length, num_channels)
        x_input = x
    else:
        raise ValueError("Input must be (batch_size, sequence_length, num_channels) dimensional")

    if args.augmentation_ratio > 0:
        augmentation_tags = "%d"%args.augmentation_ratio
        for n in range(args.augmentation_ratio):
            x_aug, augmentation_tags = augment(x_input, y, args)
            # print("Round %d: %s done"%(n, augmentation_tags))
        if args.extra_tag:
            augmentation_tags += "_"+args.extra_tag
    else:
        augmentation_tags = args.extra_tag

    if(len(x.shape)<3):
        # Reverse to two-dimensional in whole series augmentation scenario
        x_aug = x_aug.squeeze(0)
    return x_aug, y_aug, augmentation_tags