# Computer vision

#### 카메라 공식 

1/D' + 1/D = 1/f  , 여기서 D' == 초점이 맞는 CCD와 렌즈 간 거리, D == 물체와 렌즈 간 거리, f == focal length


#### 카메라 Equation

X_w에 R|t를 곱하면 X_c와 같이 시점을 이동할 수 있음

Zx' = K|M_o|X_w = M(PPM)|X_w  , 여기서 M_o == R|t, K == K_s|K_f로 [[fS_x S_theta O_x] [0 fS_y O_y] [0 0 1]]

이렇게 구해진 Zx'은 완벽한 CCD상의 좌표는 아님 왜냐하면 CCD상에는 Z좌표가 없으므로 Z로 나누어줘야함.

x' = (x,y,1)  x == u/w , y == v/w  , 여기서 w가 위에서 Z


#### 카메라 캘리브레이션 : PPM을 구하는 것

PPM의 미지수는 12개 즉, equation 12개 있어야 구할 수 있음 

하나의 좌표당 equation 2개나옴. Am=0 꼴 여기서 m은 eigenvector of A_T|A with the smallest eigenvalue임

이렇게 M(PPM)을 구했으면 그 안에서 M_int, M_ext는 QR decomposition으로 분리해서 구함. 이때 Q : orthonomal matrix, R : upper triangle matrix

Q <--> Rotation에 매칭, R <--> K(K_s|K_f)에 매칭됌

