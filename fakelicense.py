import os
import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw
import random

class yellowDraw:
    def __init__(self):
        self._font = [
        ImageFont.truetype(os.path.join(os.path.dirname(__file__), "resource/eng_92.ttf"), 126),
        ImageFont.truetype(os.path.join(os.path.dirname(__file__), "resource/zh_cn_92.ttf"), 95)
    ]
        self._bg = cv2.resize(cv2.imread(os.path.join(os.path.dirname(__file__), "resource/yellow_bg.png")), (440, 140))

    def __call__(self, plate):
        if len(plate) != 7:
            print("ERROR: Invalid length")
            return None
        fg = self._draw_fg(plate)
        return cv2.cvtColor(cv2.bitwise_and(fg, self._bg), cv2.COLOR_BGR2RGB)

    def _draw_char(self, ch):
        img = Image.new("RGB", (45 if ch.isupper() or ch.isdigit() else 95, 140), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text(
            (0, -11 if ch.isupper() or ch.isdigit() else 3), ch,
            fill = (0, 0, 0),
            font = self._font[0 if ch.isupper() or ch.isdigit() else 1]
        )
        if img.width > 45:
            img = img.resize((45, 140))
        return np.array(img)

    def _draw_fg(self, plate):
        img = np.array(Image.new("RGB", (440, 140), (255, 255, 255)))
        offset = 15
        img[0:140, offset:offset+45] = self._draw_char(plate[0])
        offset = offset + 45 + 12
        img[0:140, offset:offset+45] = self._draw_char(plate[1])
        offset = offset + 45 + 34
        for i in range(2, len(plate)):
            img[0:140, offset:offset+45] = self._draw_char(plate[i])
            offset = offset + 45 + 12
        return img


class blueDraw:
    def __init__(self):
        self._font = [
        ImageFont.truetype(os.path.join(os.path.dirname(__file__), "resource/eng_92.ttf"), 126),
        ImageFont.truetype(os.path.join(os.path.dirname(__file__), "resource/zh_cn_92.ttf"), 95)
    ]
        self._bg = cv2.resize(cv2.imread(os.path.join(os.path.dirname(__file__), "resource/blue_bg.png")), (440, 140))

    def __call__(self, plate):
        if len(plate) != 7:
            print("ERROR: Invalid length")
            return None
        fg = self._draw_fg(plate)
        return cv2.cvtColor(cv2.bitwise_or(fg, self._bg), cv2.COLOR_BGR2RGB)

    def _draw_char(self, ch):
        img = Image.new("RGB", (45 if ch.isupper() or ch.isdigit() else 95, 140), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.text(
            (0, -11 if ch.isupper() or ch.isdigit() else 3), ch,
            fill = (255, 255, 255),
            font = self._font[0 if ch.isupper() or ch.isdigit() else 1]
        )
        if img.width > 45:
            img = img.resize((45, 140))
        return np.array(img)

    def _draw_fg(self, plate):
        img = np.array(Image.new("RGB", (440, 140), (0, 0, 0)))
        offset = 15
        img[0:140, offset:offset+45] = self._draw_char(plate[0])
        offset = offset + 45 + 12
        img[0:140, offset:offset+45] = self._draw_char(plate[1])
        offset = offset + 45 + 34
        for i in range(2, len(plate)):
            img[0:140, offset:offset+45] = self._draw_char(plate[i])
            offset = offset + 45 + 12
        return img

class blackDraw:
    def __init__(self):
        self._font = [
        ImageFont.truetype(os.path.join(os.path.dirname(__file__), "resource/eng_92.ttf"), 126),
        ImageFont.truetype(os.path.join(os.path.dirname(__file__), "resource/zh_cn_92.ttf"), 95)
    ]
        self._bg = cv2.resize(cv2.imread(os.path.join(os.path.dirname(__file__), "resource/black_bg.png")), (440, 140))

    def __call__(self, plate):
        if len(plate) != 7:
            print("ERROR: Invalid length")
            return None
        fg = self._draw_fg(plate)
        return cv2.cvtColor(cv2.bitwise_or(fg, self._bg), cv2.COLOR_BGR2RGB)

    def _draw_char(self, ch):
        img = Image.new("RGB", (45 if ch.isupper() or ch.isdigit() else 95, 140), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.text(
            (0, -11 if ch.isupper() or ch.isdigit() else 3), ch,
            fill = (255, 255, 255),
            font = self._font[0 if ch.isupper() or ch.isdigit() else 1]
        )
        if img.width > 45:
            img = img.resize((45, 140))
        return np.array(img)
    def _draw_fg(self, plate):
        img = np.array(Image.new("RGB", (440, 140), (0, 0, 0)))
        offset = 15
        img[0:140, offset:offset+45] = self._draw_char(plate[0])
        offset = offset + 45 + 12
        img[0:140, offset:offset+45] = self._draw_char(plate[1])
        offset = offset + 45 + 34
        for i in range(2, len(plate)):
            img[0:140, offset:offset+45] = self._draw_char(plate[i])
            offset = offset + 45 + 12
        return img
    
class greenDraw:
    def __init__(self) -> None:
        self._font = self.load_font()
        self._bg = [
        cv2.resize(cv2.imread(os.path.join(os.path.dirname(__file__), "resource/green_bg_0.png")), (480, 140)),
        cv2.resize(cv2.imread(os.path.join(os.path.dirname(__file__), "resource/green_bg_1.png")), (480, 140))
    ]

    def __call__(self, plate, bg=0):
        if len(plate) != 8:
            print("ERROR: Invalid length")
            return None
        try:
            fg = self._draw_fg(plate)
            return cv2.cvtColor(cv2.bitwise_and(fg, self._bg[bg]), cv2.COLOR_BGR2RGB)
        except KeyError:
            print("ERROR: Invalid character")
            return None
        except IndexError:
            print("ERROR: Invalid background index")
            return None

    def _draw_char(self, ch):
        return cv2.resize(self._font[ch], (43 if ch.isupper() or ch.isdigit() else 45, 90))

    def _draw_fg(self, plate):
        img = np.array(Image.new("RGB", (480, 140), (255, 255, 255)))
        offset = 15
        img[25:115, offset:offset+45] = self._draw_char(plate[0])
        offset = offset + 45 + 9
        img[25:115, offset:offset+43] = self._draw_char(plate[1])
        offset = offset + 43 + 49
        for i in range(2, len(plate)):
            img[25:115, offset:offset+43] = self._draw_char(plate[i])
            offset = offset + 43 + 9
        return img
    def load_font(self):
        province = ["京", "津", "冀", "晋", "蒙","辽","吉","黑","沪","苏","浙","皖",
        "闽","赣","鲁","豫","鄂","湘","粤","桂","琼","渝","川","贵","云","藏","陕","甘","青","宁","新"]
        alpha = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        dict1 = {province[i]:cv2.imread(os.path.join(os.path.dirname(__file__), "resource/ne"+str(i).zfill(3)+".png" )) 
                for i in range(len(province))}
        dict2 = {alpha[i]:cv2.imread(os.path.join(os.path.dirname(__file__), "resource/ne1"+str(i).zfill(2)+".png"))
                for i in range(len(alpha))}
        dict1.update(dict2)
        return dict1

class Draw:
    _draw = [
        blackDraw(),
        blueDraw(),
        yellowDraw(),
        greenDraw()
    ]
    _provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新"]
    _alphabets = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
    _ads = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

    def __call__(self):
        draw = random.choice(self._draw)
        candidates = [self._provinces, self._alphabets]
        if type(draw) == greenDraw:
            candidates += [self._ads] * 6
            label = "".join([random.choice(c) for c in candidates])
            return draw(label, random.randint(0, 1)), label
            
        elif type(draw) == blackDraw:
            if random.random() < 0.3:
                candidates += [self._ads] * 4
                candidates += [["港", "澳"]]
            else:
                candidates += [self._ads] * 5
            label = "".join([random.choice(c) for c in candidates])
            return draw(label), label
        elif type(draw) == yellowDraw:
            if random.random() < 0.3:
                candidates += [self._ads] * 4
                candidates += [["学"]]
            else:
                candidates += [self._ads] * 5
            label = "".join([random.choice(c) for c in candidates])
            return draw(label), label
        else:
            candidates += [self._ads] * 5
            label = "".join([random.choice(c) for c in candidates])
            return draw(label), label


if __name__ == "__main__":
    import math
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="Generate a green plate.")
    parser.add_argument("--num", help="set the number of plates (default: 9)", type=int, default=12)
    args = parser.parse_args()

    draw = Draw()
    rows = math.ceil(args.num / 3)
    cols = min(args.num, 3)
    plt.figure(dpi=500)
    for i in range(args.num):
        plate, label = draw()
        print(label)
        plt.subplot(rows, cols, i + 1)
        plt.imshow(plate)
        plt.axis("off")
    plt.savefig('fakelicense',bbox_inches='tight')
    #plt.show()
