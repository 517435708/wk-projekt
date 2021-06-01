import cv2
import numpy as np
import matplotlib.pyplot as plt


def dummy(s, t = 'xD'):
    return s


imshow = dummy

name = 'billard.mp4'

cap = cv2.VideoCapture(name)
ret, frame = cap.read()

frame_for_black_bile = None

for i in range(100):
  _, frame_for_black_bile = cap.read()

cap.release()

img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

imshow('', frame)
imshow('', img_gray)

kernel = np.ones((15, 15), np.float32) / 225


frame_where_we_cut_template_black = cv2.filter2D(frame_for_black_bile, -1, kernel)
frame_where_we_cut_template = cv2.filter2D(frame, -1, kernel)
imshow('', frame_where_we_cut_template)

pts1 = np.float32([[100, 100], [2220, 110], [95, 1150], [2220, 1150]])  # 4 corners points of ORIGINAL image
pts2 = np.float32([[0, 0], [2200, 0], [0, 1200], [2200, 1200]])

matrix = cv2.getPerspectiveTransform(pts1, pts2)  # getting perspective by 4 points of each image


def transform(z):
    return cv2.warpPerspective(z, matrix, (2200, 1200))  # warps perpective to new image


template_red_bile = frame_where_we_cut_template[525:570, 463:508]  # 45 / 45
imshow('', template_red_bile)

template_white_bile = frame_where_we_cut_template[505:550, 285:330]
imshow('', template_white_bile)

template_funny_bile = frame_where_we_cut_template[595:640, 645:690]
imshow('', template_funny_bile)

template_yellow_bile = frame_where_we_cut_template[422:467, 1758:1803]
imshow('', template_yellow_bile)

template_blue_bile = frame_where_we_cut_template[595:640, 1135:1180]
imshow('', template_blue_bile)

template_green_bile = frame_where_we_cut_template[770:815, 1760:1805]
imshow('', template_green_bile)

template_black_bile = frame_where_we_cut_template_black[590:635, 263:308] #45 / 45
imshow(template_black_bile)


def prepare_frame(non_prepared):
    return cv2.filter2D(non_prepared, -1, kernel)
    # return cv2.filter2D(prepared, -1, sharp_kernel)


def get_eroded_frame_with_bile(prepared_frame, template, tresh):
    img = prepared_frame.copy()
    res = cv2.matchTemplate(img, template, cv2.TM_SQDIFF_NORMED)
    res = res * 255
    res = res.astype(np.uint8)
    res = 255 - res
    _, thresh = cv2.threshold(res, tresh, 255, cv2.THRESH_BINARY_INV)
    kernel = np.ones((7, 7), np.uint8)
    erosion = cv2.erode(thresh, kernel, iterations=3)
    return res, 255 - erosion


def black_bile(prepared_frame):
    return  get_eroded_frame_with_bile(prepared_frame, template_black_bile, 245)

def red_bile(prepared_frame):
    return get_eroded_frame_with_bile(prepared_frame, template_red_bile, 228)


def white_bile(prepared_frame):
    return get_eroded_frame_with_bile(prepared_frame, template_white_bile, 241)


def yellow_bile(prepared_frame):
    return get_eroded_frame_with_bile(prepared_frame, template_yellow_bile, 237)


def blue_bile(prepared_frame):
    return get_eroded_frame_with_bile(prepared_frame, template_blue_bile, 230)


def green_bile(prepared_frame):
    return get_eroded_frame_with_bile(prepared_frame, template_green_bile, 245)


def funny_bile(prepared_frame):
    return get_eroded_frame_with_bile(prepared_frame, template_funny_bile, 246)


def prepare_frem_for_object_detection(prepared_frame):
    res_red, red_frame = red_bile(prepared_frame)
    res_white, white_frame = white_bile(prepared_frame)
    res_yellow, yellow_frame = yellow_bile(prepared_frame)
    res_blue, blue_frame = blue_bile(prepared_frame)
    #res_green, green_frame = green_bile(prepared_frame)
    res_funny, funny_frame = funny_bile(prepared_frame)
    res_black, black_frame = black_bile(prepared_frame)
    # print(red_frame.shape, white_frame.shape, yellow_frame.shape, blue_frame.shape, green_frame.shape, funny_frame.shape, sep='|')
    combined = red_frame + white_frame + yellow_frame + blue_frame + funny_frame + black_frame#green_frame + funny_frame
    return [red_frame, white_frame, yellow_frame, blue_frame, funny_frame, black_frame], combined


def compare_frames(frame1, frame2):
    f1rgb = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
    f2rgb = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.imshow('', f1rgb)
    plt.title('first frame')

    plt.subplot(1, 2, 2)
    plt.imshow('', f2rgb)
    plt.title('result of transformation')
    plt.axis('off')
    plt.show()


def find_centers_of_circles(img):
    # gray_img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2GRAY)
    # ret, thresh = cv2.threshold(grayImage, 250, 255, cv2.THRESH_BINARY)

    # h, w = np.shape(thresh)
    # thresh[0:2,0:2] = 0.0
    # thresh[h-2:h, 0:2] = 0.0
    # thresh[0:2,w-2:w] = 0.0
    # thresh[h-2:h, w-2:w] = 0.0

    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = []
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] == 0.0 == 0:
            continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        cnts.append([cX, cY])

    final_cnts = np.array(cnts)

    return final_cnts


def draw_bbox(frame, bbox, color=(255, 255, 255)):
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
    cv2.rectangle(frame, p1, p2, color, 1, 1)


def add_rectangles_to_balls(img, biles, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    fontColor = (124, 122, 12)
    lineType = 2

    print(text, len(biles))


    for bile in biles:
        if not bile.deactivated:
            draw_bbox(img, (bile.x - (bile.width * bile.height * 0.25) ** (0.5) + bile.offset_x,
                            bile.y + bile.offset_y - (bile.width * bile.height * 0.25) ** (0.5), bile.width,
                            bile.height))
            cv2.putText(img, str(text),
                        (bile.x, bile.y),
                        font,
                        fontScale,
                        fontColor,
                        lineType)


class Bile:
    def __init__(self, x, y, id):
        self.id = id
        self.x = x
        self.y = y
        self.width = 45
        self.height = 45
        self.deactivated = False
        self.colide_with = []
        self.offset_x = 23
        self.offset_y = 23
        self.is_moving = False

    def colide(self, bile):
        if self.x - self.width < bile.x and self.x + self.width > bile.x:
            if self.y - self.height + self.offset_y < bile.y and self.y + self.height > bile.y:
                return True
        return False

    def update(self, center):
        self.deactivated = False
        self.is_moving = True
        self.x = center[0]
        self.y = center[1]

    def update_colision(self, bile, colision):
        if colision:
            self.colide_with.append(bile)
        elif bile in self.colide_with:
            self.colide_with.remove(bile)

    def has_colision(self, bile):
        for colided in self.colide_with:
            if colided == bile:
                return True
        return False


from math import sqrt


def update_biles_in_frame(biles, centers):
    min_dist = 10000
    min_center = 0
    for bile in biles:
        min_dist = 10000
        min_center = 0
        for i, center in enumerate(centers):
            dist = get_distance((bile.x, bile.y), center)
            if dist < min_dist:
                min_dist = dist
                min_center = i
        if min_dist < bile.width + 5:
            bile.update(centers[min_center])


def count_low_distances(dists):
    counter = 0
    if len(dists) == 0:
        return 1000
    for d in dists:
        if d <= 48:
            counter += 1
    return counter


def compare_distances(f_d, o_d):
    fresh_d = count_low_distances(f_d)
    old_d = count_low_distances(o_d)
    print('fresh :', fresh_d, 'old : ', old_d)
    print(f_d)
    print(o_d)
    return max(0, fresh_d - old_d)


def get_distances(biles):
    distances = []
    line = []
    for b1 in biles:
        for b2 in biles:
            if not (b1.x == b2.x or b1.y == b2.y):
                distance = get_distance((b1.x, b1.y), (b2.x, b2.y))
                if distance < 150:
                    distances.append(int(distance))
                    line.append((int((b1.x + b2.x)/2), int((b1.y + b2.y)/2), int(distance)))

    return distances, line


def get_distance(p1, p2):
    dist = sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    return dist


def draw_distances(img, lines):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (20, 20)
    fontScale = 1
    fontColor = (200, 22, 62)
    lineType = 2

    for line in lines:
        cv2.putText(img, str(line[2]), (line[0], line[1]), font, fontScale, fontColor, lineType)


def draw_additional_data(img, hit, movement):
    print(movement, '<=movement')
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (400, 20)
    fontScale = 1
    fontColor = (220, 20, 60)
    lineType = 2

    text = f'Is white bile moving: {movement}. Does white bile hit smth in this movement: {hit}'
    cv2.putText(img, str(text),
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)



def draw_num_of_collisions(img, num):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (20, 20)
    fontScale = 1
    fontColor = (20, 20, 220)
    lineType = 2

    text = f'wykryto {num} zderzen'
    cv2.putText(img, str(text),
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)


cap2 = cv2.VideoCapture(name)

shape_of_video = transform(frame).shape
w, h = shape_of_video[1], shape_of_video[0]

cbv = cv2.VideoWriter('./combined.mp4', cv2.VideoWriter_fourcc(*'DIVX'), cap2.get(cv2.CAP_PROP_FPS), (2156, 1156))
wbv = cv2.VideoWriter('./wb.mp4', cv2.VideoWriter_fourcc(*'DIVX'), cap2.get(cv2.CAP_PROP_FPS), (2156, 1156))
rbv = cv2.VideoWriter('./rb.mp4', cv2.VideoWriter_fourcc(*'DIVX'), cap2.get(cv2.CAP_PROP_FPS), (2156, 1156))
ybv = cv2.VideoWriter('./yb.mp4', cv2.VideoWriter_fourcc(*'DIVX'), cap2.get(cv2.CAP_PROP_FPS), (2156, 1156))
bbv = cv2.VideoWriter('./bb.mp4', cv2.VideoWriter_fourcc(*'DIVX'), cap2.get(cv2.CAP_PROP_FPS), (2156, 1156))
kbv = cv2.VideoWriter('./kb.mp4', cv2.VideoWriter_fourcc(*'DIVX'), cap2.get(cv2.CAP_PROP_FPS), (2156, 1156))
fbv = cv2.VideoWriter('./fb.mp4', cv2.VideoWriter_fourcc(*'DIVX'), cap2.get(cv2.CAP_PROP_FPS), (2156, 1156))

output = cv2.VideoWriter('./output.mp4', cv2.VideoWriter_fourcc(*'DIVX'), cap2.get(cv2.CAP_PROP_FPS), (w, h))

xD = 1
biles_inited = False
total_colisions = 0

old_distances = []

white_bile_center = None
white_bile_hit_another_bile_in_this_movement = False
old_center_of_white_bile = None
old_count_of_bile_distances = 0


def is_white_bile_moving(f_center, o_center, o_center2):
    if o_center is None or f_center is None:
        return False
    if abs(f_center[0] - o_center[0]) > 1 or abs(f_center[1] - o_center[1]) > 1:
        return True
    if o_center2 is None:
        return False
    if abs(f_center[0] - o_center2[0]) > 1 or abs(f_center[1] - o_center2[1]) > 1:
        return True
    return False


def distances_to_other_biles(white, biles):
    distances = []
    print(white)
    for b in biles:
        if not (b.x == white[0] or b.y == white[1]):
            distance = get_distance((b.x, white[0]), (b.x, white[1]))
            if distance < 150:
                distances.append(int(distance))
    return distances


def did_white_bile_hit_smth(white, biles, ocb):
    fresh = count_low_distances(distances_to_other_biles(white, biles))
    if fresh > ocb:
        ocb = fresh
        return True, ocb
    ocb = fresh
    return False, ocb

wbc2 = None

while cap2.isOpened():
    xD += 1
    ret, frame = cap2.read()

    transformed = transform(frame)
    prepared = prepare_frame(transformed)
    frames, combined = prepare_frem_for_object_detection(prepared)

    white_bile_is_moving = False

    centers = []
    all_biles = []
    j = 0
    for f in frames:
        j += 1
        single_center = find_centers_of_circles(f)
        biles = []
        for i, center in enumerate(single_center):
            biles.append(Bile(center[0], center[1], i*j+j))
        if j == 1:
            add_rectangles_to_balls(transformed, biles, 'red')
        elif j == 2:
            add_rectangles_to_balls(transformed, biles, 'white')
            white_bile_is_moving = is_white_bile_moving(center, white_bile_center, wbc2)
            wbc2 = white_bile_center
            white_bile_center = center
        elif j == 3:
            add_rectangles_to_balls(transformed, biles, 'yellow')
        elif j == 4:
            add_rectangles_to_balls(transformed, biles, 'bl/gr')
        elif j == 5:
            add_rectangles_to_balls(transformed, biles, 'funny')
        elif j == 6:
            add_rectangles_to_balls(transformed, biles, 'black')
        all_biles.extend(biles)

    switch, old_count_of_bile_distances = did_white_bile_hit_smth(white_bile_center, all_biles, old_count_of_bile_distances)

    if switch:
        white_bile_hit_another_bile_in_this_movement = True

    if not white_bile_is_moving:
        white_bile_hit_another_bile_in_this_movement = False

    distances, lines = get_distances(all_biles)
    colisions = int(compare_distances(distances, old_distances) / 2)

    print(colisions)
    old_distances = distances
    total_colisions += colisions

    draw_num_of_collisions(transformed, total_colisions)
    draw_additional_data(transformed, white_bile_hit_another_bile_in_this_movement, white_bile_is_moving)
    draw_distances(combined, lines)

    output.write(transformed)

    cbv.write(cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR))
    rbv.write(cv2.cvtColor(frames[0], cv2.COLOR_GRAY2BGR))
    wbv.write(cv2.cvtColor(frames[1], cv2.COLOR_GRAY2BGR))
    ybv.write(cv2.cvtColor(frames[2], cv2.COLOR_GRAY2BGR))
    bbv.write(cv2.cvtColor(frames[3], cv2.COLOR_GRAY2BGR))
    fbv.write(cv2.cvtColor(frames[4], cv2.COLOR_GRAY2BGR))
    kbv.write(cv2.cvtColor(frames[5], cv2.COLOR_GRAY2BGR))
    print('frame', xD)

    if xD > 1188: #put it to 1188
        break


kbv.release()
cbv.release()
output.release()
rbv.release()
wbv.release()
ybv.release()
bbv.release()
#gbv.release()
cap2.release()

print('Done')