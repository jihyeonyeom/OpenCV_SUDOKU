import cv2
import numpy as np
import easyocr
import sys
import copy

# 관심 영역 설정
def drawROI(img, corners):
    cpy = img.copy()

    c1 = (192, 192, 255)
    c2 = (128, 128, 255)

    for pt in corners:
        cv2.circle(cpy, tuple(pt.astype(int)), 25, c1, -1, cv2.LINE_AA)

    cv2.line(cpy, tuple(corners[0].astype(int)), tuple(corners[1].astype(int)), c2, 2, cv2.LINE_AA)
    cv2.line(cpy, tuple(corners[1].astype(int)), tuple(corners[2].astype(int)), c2, 2, cv2.LINE_AA)
    cv2.line(cpy, tuple(corners[2].astype(int)), tuple(corners[3].astype(int)), c2, 2, cv2.LINE_AA)
    cv2.line(cpy, tuple(corners[3].astype(int)), tuple(corners[0].astype(int)), c2, 2, cv2.LINE_AA)

    disp = cv2.addWeighted(img, 0.3, cpy, 0.7, 0)

    return disp

# 마우스 이벤트
def onMouse(event, x, y, flags, param):
    global srcQuad, dragSrc, ptOld, src

    if event == cv2.EVENT_LBUTTONDOWN:
        for i in range(4):
            if cv2.norm(srcQuad[i] - (x, y)) < 25:
                dragSrc[i] = True
                ptOld = (x, y)
                break

    if event == cv2.EVENT_LBUTTONUP:
        for i in range(4):
            dragSrc[i] = False

    if event == cv2.EVENT_MOUSEMOVE:
        for i in range(4):
            if dragSrc[i]:
                dx = x - ptOld[0]
                dy = y - ptOld[1]

                srcQuad[i] += (dx, dy)

                cpy = drawROI(src, srcQuad)
                cv2.imshow('img', cpy)
                ptOld = (x, y)
                break

# 입력 이미지 불러오기
src = cv2.imread('sudoku4.jpg')
src = cv2.resize(src, (550, 550))

if src is None:
    print('Image open failed!')
    sys.exit()
    
# 입력 영상 크기 및 출력 영상 크기
h, w = src.shape[:2]
dw = 500
dh = round(dw * 297 / 210)  # A4 용지 크기: 210x297cm

# 모서리 점들의 좌표, 드래그 상태 여부
srcQuad = np.array([[30, 30], [30, h-30], [w-30, h-30], [w-30, 30]], np.float32)
dstQuad = np.array([[0, 0], [0, dh-1], [dw-1, dh-1], [dw-1, 0]], np.float32)
dragSrc = [False, False, False, False]

# 모서리점, 사각형 그리기
disp = drawROI(src, srcQuad)

cv2.imshow('img', disp)
cv2.setMouseCallback('img', onMouse)

while True:
    key = cv2.waitKey()
    if key == 13:  # ENTER 키
        break
    elif key == 27:  # ESC 키
        cv2.destroyAllWindows()
        sys.exit()

# 투시 변환
pers = cv2.getPerspectiveTransform(srcQuad, dstQuad)
dst = cv2.warpPerspective(src, pers, (dw, dh), flags=cv2.INTER_CUBIC)
dst = cv2.resize(dst, (450, 450))

# 사진에서 추출 받을 스도쿠 배열 0으로 초기화
sudoku_board = [[0 for _ in range(9)] for _ in range(9)]
# 정답 값 저장 리스트
sudoku_answer = []

# 숫자 검출
reader = easyocr.Reader(['ko'], gpu=False)
box_size = 50
for row in range(9):
    for col in range(9):
        dst_temp = dst[row * box_size : (row + 1) * box_size, col * box_size : (col + 1) * box_size]
        dst_temp = cv2.resize(dst_temp, (150, 150))
        dst_temp = cv2.cvtColor(dst_temp, cv2.COLOR_BGR2GRAY)
        # 이미지에서 숫자값 추출
        result = reader.readtext(dst_temp, detail=0, allowlist="0123456789")
        # 찾아낸 값이 있을 경우 해당 값 해당 위치에 저장
        if result:
            cv2.imshow('dst_temp', dst_temp)
            cv2.waitKey()
            number = int(result[0])
            sudoku_board[row][col] = number

# 스도쿠 n 값이 타당한지 체크
def sudoku_check(x, y, n):
    # 행 체크
    if n in sudoku_board[x]:
        return False
    # 열 체크
    for i in range(9):
        if sudoku_board[i][y] == n:
            return False
    # 3x3 박스 체크
    nx = (x // 3) * 3
    ny = (y // 3) * 3
    for i in range(nx, nx + 3):
        for j in range(ny, ny + 3):
            if sudoku_board[i][j] == n:
                return False
    return True

# 스도쿠 문제 해결 메서드
def sudoku_dfs_solve(x, y):
    global sudoku_board, sudoku_answer, dst
    # 다음 행으로 넘기기
    if y == 9:
        x, y = x + 1, 0
    # 행이 9가 되면 종료
    if x == 9:
        sudoku_answer = np.array(sudoku_board)
        return
    # 0인 인덱스 값 찾기
    if sudoku_board[x][y] == 0:
        # 1부터 9까지의 값 대입
        for i in range(1, 10):
            # 스도쿠 규칙 기준으로 가능한지 체크
            if sudoku_check(x, y, i):
                # 해당 인덱스에 대입
                sudoku_board[x][y] = i
                # 대입 전 상태로 돌아가기 위한 카피
                dst_temp = copy.deepcopy(dst)
                # 대입한 숫자 이미지에 나타내기
                cv2.putText(dst, str(i), (y * box_size + 15, (x + 1) * box_size - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.imshow('dst', dst)
                cv2.waitKey(1)
                # 깊이 우선 탐색으로 모든 경우의 수 돌아보기
                sudoku_dfs_solve(x, y + 1)
                # 해당 값이 틀린 경우 0으로 되돌리기
                sudoku_board[x][y] = 0
                # 이미지 대입 전 상태로 돌리기
                dst = copy.deepcopy(dst_temp)
                cv2.imshow('dst', dst)
                cv2.waitKey(1)
    else:
        sudoku_dfs_solve(x, y + 1)
    return
    
sudoku_dfs_solve(0, 0)

#스도쿠 정답 출력
print(sudoku_answer)
for row in range(9):
    for col in range(9):
        dst_temp = dst[row * box_size : (row + 1) * box_size, col * box_size : (col + 1) * box_size]
        dst_temp = cv2.resize(dst_temp, (150, 150))
        dst_temp = cv2.cvtColor(dst_temp, cv2.COLOR_BGR2GRAY)
        result = reader.readtext(dst_temp, detail=0, allowlist="0123456789")
        # 찾아낸 값이 없을 경우 빈칸에 정답 출력
        if not result:
            cv2.putText(dst, str(sudoku_answer[row][col]), (col * box_size + 15, (row + 1) * box_size - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.imshow('dst', dst)
            cv2.waitKey(25)
cv2.imshow('dst', dst) 
cv2.waitKey()
cv2.destroyAllWindows()
