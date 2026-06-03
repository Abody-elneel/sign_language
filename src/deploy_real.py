import pickle, json, time
import numpy as np
import cv2
import mediapipe as mp

BaseOptions       = mp.tasks.BaseOptions
HandLandmarker    = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# BGR colors
CYAN    = (255, 255, 0)
GREEN   = (57, 255, 20)
ORANGE  = (0, 165, 255)
PINK    = (180, 0, 255)
FLASH   = (0, 220, 100)
DIM     = (100, 100, 140)
BORDER  = (40, 40, 70)
BLACK   = (0, 0, 0)
WHITE   = (255, 255, 255)

FINGER_COLORS = [(255,80,80),(80,200,255),(80,255,140),(255,200,80),(200,80,255)]
FINGER_SEGS   = [[(0,1),(1,2),(2,3),(3,4)],[(0,5),(5,6),(6,7),(7,8)],
                 [(0,9),(9,10),(10,11),(11,12)],[(0,13),(13,14),(14,15),(15,16)],
                 [(0,17),(17,18),(18,19),(19,20)]]
PALM_CONNS    = [(5,9),(9,13),(13,17)]


def glow_line(img, p1, p2, col, t=2):
    cv2.line(img, p1, p2, tuple(int(c*0.2) for c in col), t+4)
    cv2.line(img, p1, p2, col, t)

def glow_circle(img, c, r, col):
    cv2.circle(img, c, r+3, tuple(int(x*0.2) for x in col), 1)
    cv2.circle(img, c, r, col, -1)

def draw_landmarks(rgb, result):
    out = np.copy(rgb)
    h, w = out.shape[:2]
    for lms in result.hand_landmarks:
        for fi, segs in enumerate(FINGER_SEGS):
            for a, b in segs:
                p1 = (int(lms[a].x*w), int(lms[a].y*h))
                p2 = (int(lms[b].x*w), int(lms[b].y*h))
                glow_line(out, p1, p2, FINGER_COLORS[fi])
        for a, b in PALM_CONNS:
            glow_line(out, (int(lms[a].x*w),int(lms[a].y*h)),
                           (int(lms[b].x*w),int(lms[b].y*h)), CYAN, 1)
        for i, lm in enumerate(lms):
            x, y = int(lm.x*w), int(lm.y*h)
            col = FINGER_COLORS[min(i//4,4)]
            if i in [4,8,12,16,20]:
                glow_circle(out, (x,y), 6, col)
                cv2.circle(out, (x,y), 3, WHITE, -1)
            else:
                cv2.circle(out, (x,y), 4, col, -1)
    return out

def brackets(img, x, y, bw, bh, col, s=16):
    for px, py_, dx, dy in [(x,y,s,0),(x,y,0,s),(x+bw,y,-s,0),(x+bw,y,0,s),
                             (x,y+bh,s,0),(x,y+bh,0,-s),(x+bw,y+bh,-s,0),(x+bw,y+bh,0,-s)]:
        cv2.line(img, (px,py_), (px+dx,py_+dy), col, 2)

def shadow_text(img, txt, pos, scale, col, t=1):
    cv2.putText(img, txt, (pos[0]+2,pos[1]+2), cv2.FONT_HERSHEY_SIMPLEX, scale,
                tuple(max(0,c-150) for c in col), t+1)
    cv2.putText(img, txt, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, col, t)


def main():
    with open('label_mapping.json', encoding='utf-8') as f:
        label_map = json.load(f)
    model = pickle.load(open('hand_gesture_model.pkl', 'rb'))

    opts = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
        running_mode=VisionRunningMode.VIDEO, num_hands=2)

    WW, WH = 1280, 720
    CW, CH, CX, CY = 840, 630, 20, 55
    PX = CX + CW + 20
    PW = WW - PX - 20

    cv2.namedWindow('Hand Gesture Recognition', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Hand Gesture Recognition', WW, WH)

    with HandLandmarker.create_from_options(opts) as lmk:
        cap = cv2.VideoCapture(0)
        cap.set(3, 1280); cap.set(4, 720)

        word, last_letter = "", ""
        flash_t = 0.0
        FLASH_DUR = 0.6
        frame_n = 0
        status = "Press  P  to Predict"
        history, MAX_H = [], 8

        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ts    = int(time.time() * 1000)
            res   = lmk.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
            has_hand = bool(res.hand_landmarks)

            # Canvas
            canvas = np.full((WH, WW, 3), (10,10,18), dtype=np.uint8)

            # Camera
            ann = cv2.cvtColor(draw_landmarks(rgb, res), cv2.COLOR_RGB2BGR)
            cam = cv2.resize(ann, (CW, CH))
            if not has_hand:
                cv2.addWeighted(cam, 0.65, np.zeros_like(cam), 0.35, 0, cam)
            # Scan line
            sy = int((frame_n * 3) % CH)
            ov = cam.copy(); cv2.line(ov,(0,sy),(CW,sy),(0,255,100),1)
            cv2.addWeighted(ov, 0.08, cam, 0.92, 0, cam)
            canvas[CY:CY+CH, CX:CX+CW] = cam

            bc = GREEN if has_hand else BORDER
            cv2.rectangle(canvas, (CX-1,CY-1),(CX+CW+1,CY+CH+1), bc, 1)
            brackets(canvas, CX-3, CY-3, CW+6, CH+6, bc)

            # Title
            shadow_text(canvas, "HAND GESTURE RECOGNITION", (CX, 38), 0.7, CYAN, 2)
            dc = GREEN if has_hand else (80,80,80)
            cv2.circle(canvas, (CX+CW-14, 28), 6, dc, -1)
            cv2.putText(canvas, "HAND DETECTED" if has_hand else "NO HAND",
                        (CX+CW-130,33), cv2.FONT_HERSHEY_SIMPLEX, 0.38, dc, 1)

            # Panel bg
            ov2 = canvas.copy()
            cv2.rectangle(ov2,(PX,CY),(PX+PW,CY+CH),(18,18,30),-1)
            cv2.addWeighted(ov2,0.7,canvas,0.3,0,canvas)
            cv2.rectangle(canvas,(PX,CY),(PX+PW,CY+CH),BORDER,1)
            brackets(canvas, PX, CY, PW, CH, CYAN, 12)
            cv2.putText(canvas,"OUTPUT PANEL",(PX+10,CY+26),cv2.FONT_HERSHEY_SIMPLEX,0.5,CYAN,1)
            cv2.line(canvas,(PX+10,CY+34),(PX+PW-10,CY+34),BORDER,1)

            # Word box
            WBX,WBY,WBW,WBH = PX+10, CY+58, PW-20, 52
            cv2.putText(canvas,"WORD",(WBX,WBY-10),cv2.FONT_HERSHEY_SIMPLEX,0.38,DIM,1)
            cv2.rectangle(canvas,(WBX,WBY),(WBX+WBW,WBY+WBH),(25,25,45),-1)
            cv2.rectangle(canvas,(WBX,WBY),(WBX+WBW,WBY+WBH),BORDER,1)
            flash = (time.time()-flash_t) < FLASH_DUR
            wc = FLASH if flash else WHITE
            disp = word or "_"
            fs = 1.5
            while fs > 0.5:
                (tw,_),_ = cv2.getTextSize(disp, cv2.FONT_HERSHEY_DUPLEX, fs, 2)
                if tw < WBW-16: break
                fs -= 0.1
            shadow_text(canvas, disp, (WBX+10, WBY+38), fs, wc, 2)
            if int(time.time()*2)%2==0:
                (tw,_),_=cv2.getTextSize(disp,cv2.FONT_HERSHEY_DUPLEX,fs,2)
                cv2.line(canvas,(WBX+10+tw+4,WBY+12),(WBX+10+tw+4,WBY+WBH-8),CYAN,2)

            # Last predicted
            LPY = WBY + WBH + 24
            cv2.putText(canvas,"LAST PREDICTED",(PX+10,LPY),cv2.FONT_HERSHEY_SIMPLEX,0.35,DIM,1)
            cv2.rectangle(canvas,(PX+10,LPY+6),(PX+PW-10,LPY+52),(25,25,45),-1)
            cv2.rectangle(canvas,(PX+10,LPY+6),(PX+PW-10,LPY+52),BORDER,1)
            if last_letter:
                shadow_text(canvas, last_letter, (PX+20, LPY+44), 1.2,
                            FLASH if flash else ORANGE, 2)

            # History
            HY = LPY + 70
            cv2.putText(canvas,"HISTORY",(PX+10,HY),cv2.FONT_HERSHEY_SIMPLEX,0.35,DIM,1)
            cv2.line(canvas,(PX+10,HY+6),(PX+PW-10,HY+6),BORDER,1)
            for i, e in enumerate(history[:MAX_H]):
                hy2 = HY+24+i*28
                if hy2 > CY+CH-130: break
                a = max(0.4, 1.0-i*0.08)
                cd = tuple(int(c*a) for c in DIM)
                cl = tuple(int(c*a) for c in GREEN)
                cv2.putText(canvas,f"{i+1:02d}",(PX+12,hy2),cv2.FONT_HERSHEY_SIMPLEX,0.35,cd,1)
                cv2.putText(canvas,e['l'],(PX+42,hy2),cv2.FONT_HERSHEY_SIMPLEX,0.5,cl,1)
                cv2.putText(canvas,e['t'],(PX+80,hy2),cv2.FONT_HERSHEY_SIMPLEX,0.32,cd,1)

            # Controls
            KY = CY+CH-115
            cv2.line(canvas,(PX+10,KY),(PX+PW-10,KY),BORDER,1)
            cv2.putText(canvas,"CONTROLS",(PX+10,KY+16),cv2.FONT_HERSHEY_SIMPLEX,0.35,DIM,1)
            for i,(k,desc,col) in enumerate([("P","Predict",GREEN),("D","Delete last",ORANGE),
                                              ("C","Clear all",PINK),("Q","Quit",(120,120,120))]):
                ky2 = KY+34+i*20
                cv2.rectangle(canvas,(PX+12,ky2-12),(PX+28,ky2+2),col,-1)
                cv2.putText(canvas,k,(PX+15,ky2),cv2.FONT_HERSHEY_SIMPLEX,0.38,BLACK,1)
                cv2.putText(canvas,desc,(PX+32,ky2),cv2.FONT_HERSHEY_SIMPLEX,0.35,DIM,1)

            # Status
            sc = FLASH if flash else (ORANGE if not has_hand else DIM)
            shadow_text(canvas, status, (CX, CY+CH+28), 0.55, sc)

            cv2.imshow('Hand Gesture Recognition', canvas)
            frame_n += 1

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')):
                break
            elif key in (ord('p'), ord('P')):
                if has_hand:
                    lms = res.hand_landmarks[0]
                    bx,by,bz = lms[0].x, lms[0].y, lms[0].z
                    arr = np.array([[l.x-bx,l.y-by,l.z-bz] for l in lms]).flatten().reshape(1,-1)
                    pk = str(model.predict(arr)[0])
                    if pk in label_map:
                        pl = label_map[pk]
                        word += pl; last_letter = pl
                        flash_t = time.time()
                        status = f"Predicted: '{pl}'"
                        history.insert(0, {'l': pl, 't': time.strftime('%H:%M:%S')})
                        print(f'\rWord: {word}   ', end='')
                    else:
                        status = "Unknown gesture"
                else:
                    status = "No hand detected!"
            elif key in (ord('d'), ord('D')):
                if word:
                    word = word[:-1]
                    if history: history.pop(0)
                    status = "Deleted last letter"
            elif key in (ord('c'), ord('C')):
                word = last_letter = ""
                history = []
                status = "Cleared!"

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
