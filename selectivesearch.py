import skimage.io
import skimage.feature
import skimage.color
import skimage.transform
import skimage.util
import skimage.segmentation
import numpy
import cv2

def _generate_segments(im_orig, scale, sigma, min_size):
    """
        通过Felzenswalb和Huttenlocher的算法划分最小区域
    """
    # open the Image
    im_mask = skimage.segmentation.felzenszwalb(
        skimage.util.img_as_float(im_orig), scale=scale, sigma=sigma,
        min_size=min_size)
    # 将掩码通道合并为图像作为第4通道
    im_orig = numpy.append(
        im_orig, numpy.zeros(im_orig.shape[:2])[:, :, numpy.newaxis], axis=2)
    im_orig[:, :, 3] = im_mask

    return im_orig


def _sim_colour(r1, r2):
    """
        计算直方图颜色交点的总和
    """
    return sum([min(a, b) for a, b in zip(r1["hist_c"], r2["hist_c"])])


def _sim_texture(r1, r2):
    """
        计算纹理的直方图交集的总和
    """
    return sum([min(a, b) for a, b in zip(r1["hist_t"], r2["hist_t"])])


def _sim_size(r1, r2, imsize):
    """
        计算图像的尺度相似度
    """
    return 1.0 - (r1["size"] + r2["size"]) / imsize


def _sim_fill(r1, r2, imsize):
    """
        计算图像上的填充相似度
    """
    bbsize = (
            (max(r1["max_x"], r2["max_x"]) - min(r1["min_x"], r2["min_x"]))
            * (max(r1["max_y"], r2["max_y"]) - min(r1["min_y"], r2["min_y"]))
    )
    return 1.0 - (bbsize - r1["size"] - r2["size"]) / imsize


def _calc_sim(r1, r2, imsize):
    return (_sim_colour(r1, r2) + _sim_texture(r1, r2)
            + _sim_size(r1, r2, imsize) + _sim_fill(r1, r2, imsize))


def _calc_colour_hist(img):
    """
        计算每个区域的颜色直方图
         输出直方图的大小为BINS * COLOUR_CHANNELS（3）
         bins为25，与[uijlings_ijcv2013_draft.pdf]相同
         提取HSV
    """

    BINS = 25
    hist = numpy.array([])

    for colour_channel in (0, 1, 2):
        # 提取一个颜色通道
        c = img[:, colour_channel]

        # 计算每种颜色的直方图并加入到结果中
        hist = numpy.concatenate(
            [hist] + [numpy.histogram(c, BINS, (0.0, 255.0))[0]])

    # L1正则化
    hist = hist / len(img)

    return hist


def _calc_texture_gradient(img):
    """
        计算整个图像的纹理渐变
        最初的SelectiveSearch算法为8个方向提出了高斯导数，但我们使用LBP代替。
        输出将是[height（*）] [width（*）]
    """
    ret = numpy.zeros((img.shape[0], img.shape[1], img.shape[2]))

    for colour_channel in (0, 1, 2):
        ret[:, :, colour_channel] = skimage.feature.local_binary_pattern(
            img[:, :, colour_channel], 8, 1.0)

    return ret


def _calc_texture_hist(img):
    """
        计算每个区域的纹理直方图
        计算每种颜色的梯度直方图
        输出直方图的大小为BINS * ORIENTATIONS * COLOUR_CHANNELS（3）
    """
    BINS = 10

    hist = numpy.array([])

    for colour_channel in (0, 1, 2):
        # 由颜色通道掩盖
        fd = img[:, colour_channel]

        # 计算每个方向的直方图并将它们连接起来并加入到结果中
        hist = numpy.concatenate(
            [hist] + [numpy.histogram(fd, BINS, (0.0, 1.0))[0]])

    # L1正则化
    hist = hist / len(img)

    return hist


def _extract_regions(img):
    '''
        计算像素的位置产生标记框
        计算每个区域的颜色直方图、纹理直方图、区域大小
    '''
    R = {}

    # rgb==>hsv
    hsv = skimage.color.rgb2hsv(img[:, :, :3])

    # pass 1: 计算像素位置
    for y, i in enumerate(img):

        for x, (r, g, b, l) in enumerate(i):

            # 初始化一个新的区域
            if l not in R:
                R[l] = {
                    "min_x": 0xffff, "min_y": 0xffff,
                    "max_x": 0, "max_y": 0, "labels": [l]}

            # 边框
            if R[l]["min_x"] > x:
                R[l]["min_x"] = x
            if R[l]["min_y"] > y:
                R[l]["min_y"] = y
            if R[l]["max_x"] < x:
                R[l]["max_x"] = x
            if R[l]["max_y"] < y:
                R[l]["max_y"] = y

    # pass 2: 计算纹理梯度
    tex_grad = _calc_texture_gradient(img)

    # pass 3: 计算每个区域的颜色直方图
    for k, v in R.items():
        # 颜色直方图
        masked_pixels = hsv[:, :, :][img[:, :, 3] == k]
        R[k]["size"] = len(masked_pixels / 4)
        R[k]["hist_c"] = _calc_colour_hist(masked_pixels)

        # 纹理直方图
        R[k]["hist_t"] = _calc_texture_hist(tex_grad[:, :][img[:, :, 3] == k])

    return R


def _extract_neighbours(regions):
    def intersect(a, b):
        if (a["min_x"] < b["min_x"] < a["max_x"]
            and a["min_y"] < b["min_y"] < a["max_y"]) or (
                a["min_x"] < b["max_x"] < a["max_x"]
                and a["min_y"] < b["max_y"] < a["max_y"]) or (
                a["min_x"] < b["min_x"] < a["max_x"]
                and a["min_y"] < b["max_y"] < a["max_y"]) or (
                a["min_x"] < b["max_x"] < a["max_x"]
                and a["min_y"] < b["min_y"] < a["max_y"]):
            return True
        return False

    R = regions.items()
    r = [elm for elm in R]
    R = r
    neighbours = []
    for cur, a in enumerate(R[:-1]):
        for b in R[cur + 1:]:
            if intersect(a[1], b[1]):
                neighbours.append((a, b))

    return neighbours


def _merge_regions(r1, r2):
    new_size = r1["size"] + r2["size"]
    rt = {
        "min_x": min(r1["min_x"], r2["min_x"]),
        "min_y": min(r1["min_y"], r2["min_y"]),
        "max_x": max(r1["max_x"], r2["max_x"]),
        "max_y": max(r1["max_y"], r2["max_y"]),
        "size": new_size,
        "hist_c": (
                          r1["hist_c"] * r1["size"] + r2["hist_c"] * r2["size"]) / new_size,
        "hist_t": (
                          r1["hist_t"] * r1["size"] + r2["hist_t"] * r2["size"]) / new_size,
        "labels": r1["labels"] + r2["labels"]
    }
    return rt


def selective_search(im_orig, scale=1.0, sigma=0.8, min_size=50):
    '''选择性搜索
    参数
    ----------
         im_orig：ndarray
             输入图像
         scale：int
             自由参数。 更高的scale意味着felzenszwalb分段中更大的聚类。
         sigma：float
             felzenszwalb分割的高斯核宽度。
         min_size：int
             felzenszwalb分段的最小组件大小。
    返回
    -------
         img：ndarray
             带有区域标签的图像
             区域标签存储在每个像素的第4个值[r，g，b，（区域）]
         区域：字典数组
            [
                {
                     'rect':(左上顶点x坐标，左上顶点y坐标，宽度w，高度h），
                     '标签'： [区域号]
                }，
                ...
            ]
    '''
    assert im_orig.shape[2] == 3, "3ch image is expected"

    # 加载图像并获得最小的区域，区域标签存储在每个像素的第4个值[r，g，b，（区域）]
    img = _generate_segments(im_orig, scale, sigma, min_size)

    if img is None:
        return None, {}

    imsize = img.shape[0] * img.shape[1]
    R = _extract_regions(img)   #R中保存每个区域的颜色直方图、纹理直方图、区域大小信息

    # 提取邻居信息
    neighbours = _extract_neighbours(R)

    # 计算初始相似性
    S = {}
    for (ai, ar), (bi, br) in neighbours:
        S[(ai, bi)] = _calc_sim(ar, br, imsize)

    # 分层搜索
    while S != {}:

        # 获得最高的相似度
        # i, j = sorted(S.items(), cmp=lambda a, b: cmp(a[1], b[1]))[-1][0]
        i, j = sorted(list(S.items()), key=lambda a: a[1])[-1][0]

        # 合并相应的地区
        t = max(R.keys()) + 1.0
        R[t] = _merge_regions(R[i], R[j])

        # 标记要删除的区域的相似性
        key_to_delete = []
        for k, v in S.items():
            if (i in k) or (j in k):
                key_to_delete.append(k)

        # 删除相关区域的旧相似性
        for k in key_to_delete:
            del S[k]

        # 计算与新区域的相似性集
        for k in filter(lambda a: a != (i, j), key_to_delete):
            n = k[1] if k[0] in (i, j) else k[0]
            S[(t, n)] = _calc_sim(R[t], R[n], imsize)

    regions = []
    for k, r in R.items():
        regions.append({
            'rect': (
                r['min_x'], r['min_y'],
                r['max_x'] - r['min_x'], r['max_y'] - r['min_y']),
            'size': r['size'],
            'labels': r['labels']
        })

    return img, regions

# # test--------------------
# image = cv2.imread('F:/DeepLearning/tensorflow/mnist/datas/picture/123.png')
# img, regions = selective_search(image, scale=1000, sigma=0.9, min_size=1000)
# '''
# scale:自由参数.更高的scale意味着felzenszwalb分段中更大的聚类。控制合并后区域的大小
# sigma:felzenszwalb分割的高斯核的方差。
# min_size:felzenszwalb分段的最小组件大小。分割后会有很多小区域，当区域像素点的个数小于min时，选择与其差异最小的区域合并
# '''
# print(len(regions))
# for r in regions:
#     if r['size']<=img.shape[0]*img.shape[1]:
#         x, y, w, h = r['rect']
#         cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0),2)
#         cv2.imshow('image', image)
# if cv2.waitKey(0) & 0xff == 27:
#     cv2.destroyAllWindows()
