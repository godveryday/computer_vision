# Computer vision

## Thin lens formula

1/D' + 1/D = 1/f  , 여기서 D' : 초점이 맞는 CCD와 렌즈 간 거리, D : 물체와 렌즈 간 거리, f : 광축과 평행한 빛이 렌즈를 통과한 후 한점에 맺히는데 그 점과 렌즈 간 거리

<br/>


## 카메라 Equation

X_w에 R|t를 곱하면 X_c와 같이 시점을 이동할 수 있음

Zx' = K|M_o|X_w = M(PPM)|X_w  , 여기서 M_o == R|t, K == K_s|K_f로 [[fS_x S_theta O_x] [0 fS_y O_y] [0 0 1]]

이렇게 구해진 Zx'은 완벽한 CCD상의 좌표는 아님 왜냐하면 CCD상에는 Z좌표가 없으므로 Z로 나누어줘야함.

x' = (x,y,1)  x == u/w , y == v/w  , 여기서 w가 위에서 Z

<br/>

## 카메라 캘리브레이션 : PPM을 구하는 것

PPM의 미지수는 12개 즉, equation 12개 있어야 구할 수 있음 

하나의 좌표당 equation 2개나옴. Am=0 꼴 여기서 m은 eigenvector of A_T|A with the smallest eigenvalue임

이렇게 M(PPM)을 구했으면 그 안에서 M_int, M_ext는 QR decomposition으로 분리해서 구함. 이때 Q : orthonomal matrix, R : upper triangle matrix

Q <--> Rotation에 매칭, R <--> K(K_s|K_f)에 매칭됌


### 다른 방법

PPM matrix에서 m_23을 1로 설정하면 

u_i = u/w , v_i = v/w 식에서 전개된 최종 식 Am = [u_i, v_i] 꼴로 우변이 0이 아니게 됌 

따라서 Am = b 꼴이 되고, 이때 m = (A_T|A)^-1 A_T|b --> Pseudo Inverse를 통해 구할 수 있음



<br/>

## 렌즈 왜곡

크게 2가지로 나뉨

1. Radial distortion

- barrel distortion

- Pincushion distortion


2. Tangential distortion


We have five distortion Parameters which in OpenCV (K1, K2, P1, P2, K3) <-- 왜곡 파라미터에 대해 OpenCV를 통해 확인가능함

이때 K1,K2,K3 : Radial distortion , P1, P2 : Tangential distortion 

<br/>


## Image Processing

영상 처리는 크게 2가지로 나뉨

1. Point operation

2. Area operation  == filter와 correlation, convolution을 적용


예를 들어 1-D 차원에서 생각했을 때

Original Image * h [0 0 0 1 0 0 0] == Original Image

Original Image * h [0 0 0 0 0 1 0] == 왼쪽으로 Shift (correlation인 경우)

Original Iamge * (h1 [0 0 0 2 0 0 0] - h2 [0 0 -.33 -.33 -.33 0 0]) == Sharpened original 

이를 응용한 다양한 filter가 존재함

1. A_avg == 1/9 [[1 1 1] [1 1 1] [1 1 1]]  --> 흐려짐, Average Smoothing

2. A_gaus == 1/106 [[1 1 1 1 1] [1 9 9 9 1] [1 9 18 9 1] [1 9 9 9 1] [1 1 1 1 1]] --> 기본적으로 Low-Pass Fitler

Image에 가우시안 필터를 적용해서 나온 그 이미지를 Original 이미지에 빼면 Original 이미지의 윤곽선이 남음


<br/>

### 가우시안 필터의 단점 

윤곽선도 같이 흐려짐 --> Bilateral filter를 사용하여 해결함

Gausian filtering == h[m,n] = g[k,l] f[m+k, n+l] 꼴이라면

Bilateral filter는 == h[m,n] = g[k,l] r[k,l] f[m+k, n+l] 꼴임 , 

여기서 r[k,l]이 가우시안 필터 g에 곱해지는데 r[k,l]은 두 지점의 Intensity 차이가 클수록 값이 작아짐



<br/>

## Edge detection --> 미분 사용

실수 공간에서 미분과 영상에서 미분은 큰 차이가 있음

영상에서는 극한개념이 없고, 인덱스로 사용함 따라서 미분이 단순 빼기로 바뀜

I(x+1,y) - I(x,y) 이것을 해주는 필터를 만들면 h[-1 1] (h의 첫번째인덱스를 x,y가 가리킬 때)

I(x,y+1) - I(x,y-1) 이것을 필터로 만들면 h[[-1] [0] [1]] 

주의사항) filter를 적용 시에, x축과 y축 한번에 할 수 없음

따라서, 아래와 같이 적용


### Simple edge detection Idea

1. dI/dx with h[-1 0 1] --> Resulting Image I_x

2. dI/dy with h[[-1] [0] [1]] --> Resulting Image I_y

3. I(i,j) = sqrt( I_x(i,j)^2 + I_y(i,j)^2 )

<br/>

## Laplacian Filter --> Edge 뿐 아니라 영상의 feature 찾을 때 자주 사용

원리는 Image의 Intensity가 급격히 변하는 지점을 찾는 것

라플라시안 --> 두번 미분

Original Image를 한번 미분하면 Edge부분의 값이 큼

Original Image를 두번 미분하면 Zero-crossing되는 지점이 Edge 지점임

<br/>

<img width = "55%" img src ="https://github.com/user-attachments/assets/bda30624-f32c-486a-8fbf-637adcc2fdda"> <img width = "40%" img src ="https://github.com/user-attachments/assets/7739f270-e212-47eb-8dfd-1f967de34ce2">


다만, 라플라시안 필터는 zero-crossing을 사용하기 때문에 노이즈에 민감함. 따라서 Laplacian of Gausian 이라고 하는 LOG방식을 사용 

--> 가우시안 필터를 통해 노이트를 없애고 이후 라플라시안 사용

<br/>

## Template Matching

템플릿 매칭은 내가 찾고자 하는 이미지를 찾는 것 어떻게? --> correlation 또는 단순한 빼기(SSD)를 사용

어떻게 correlation을 하면 kernel과 같은 모양을 가진 부분의 값이 크게 나올까? --> 내적처럼 동작함

<img width=50% img src="https://github.com/user-attachments/assets/609183de-a576-46cd-8ce7-4c7aae98a2a9">


<br/>

## Image Features (Harris, SIFT, HOG)  --> 여기서부터 중요

How the computer detects and tracks objects?

STOP 사인을 detection해야하는데, 

1. STOP 사인이 옆쪽으로 있다면?

2. 어두운 상황이라면?

3. STOP 사인의 크기 차이가 나는상황이라면?

--> 단순한 template matching으로는 찾기가 힘들다

이럴 때, Feature를 통해 찾음 

Good Feature란? : 영상의 변화가 많은 지점

<br/>

### Harris corner detector --> Good feature인 코너점을 찾는 알고리즘

<img width=50% img src="https://github.com/user-attachments/assets/ef72befc-2c1f-4064-9333-99ffa047b042">

위 그림처럼 코너점은 X,Y 모든 방향으로 변화가 큼

<br/><br/>

<img width=50% img src="https://github.com/user-attachments/assets/8c661988-32a7-4e91-86a5-973c09431fc7">

위에서 E(u,v)즉 SSD를 구함. 만약 코너점이라면 그 값이 커질 것임

<br/><br/>

<img width=50% img src="https://github.com/user-attachments/assets/e9521f51-f0b7-4e64-a9a3-335d2edb9086">

I(x+u, y+v)는 다음처럼 미분으로 표현할 수 있고, 테일러급수를 통해 간략화 함

<br/><br/>

<img width=50% img src="https://github.com/user-attachments/assets/fd265b98-685e-42de-8393-55e3fad05e7c"> <img width=40% img src="https://github.com/user-attachments/assets/3379cee6-fd5e-464d-844a-37c7a94293f5">

그 결과 E(u,v)는 위와같이 정리할 수 있음. 이후 옆과 같이 정리됌

여기서 M이라고 하는 행렬을 Singular value decomposition (SVD)를 통해 분해하고 

<br/><br/>

<img width=50% img src="https://github.com/user-attachments/assets/51c23e5c-a70b-49fe-9d58-a7dfc809a58a">

위 그림처럼 R이라는 스칼라값을 구함. 이는 이전 사진에서 R과 다른 R임

픽셀마다 R값을 구하고 주변보다 R값이 큰 (Local maximum of R)을 구함. 이 지점을 통해 feature matching을 진행함.

<br/><br/>

<img width=55% img src="https://github.com/user-attachments/assets/a12b0856-163d-424f-bf0d-92d2563603ed">

위 그림은 이제까지의 수학적과정을 그림으로 보여줌.

앞서 Template matching기법이 이미지의 Rotation이나, 밝기 변화 (Intensity shift)에 민감했던 반면,

Harris는 Image Scale이 변화하지 않는 이상 Intensity shift, image Rotation에는 Invarient하다 !

하지만 여전히 이미지 스케일변화에는 취약함 --> 이것을 해결해주는것이 SIFT방식이다


<br/>

### SIFT, HOG --> 나중에 정리 P10


---


## Image alignment (transformation) --> 여기부터 더욱 중요해짐

<img width=55% img src="https://github.com/user-attachments/assets/7e849c18-2109-431d-94b9-9be24120a23b">

위 사진과 같이 다양한 상황에서 Image alignment가 이뤄지고 있음 --> ADAS, 자율주행, 자율주차 등

이미지 정렬, 변환 중에서 Feature-based alignment에 대해서 학습함 --> 정렬과 변환은 비슷하지만 완전 똑같은 개념은 아님. 변환이 더 큰 개념

### Image Transformation

크게 3가지로 나누어진다

1. Euclidean Tr --> 미지수 3개

2. Affine Tr --> 미지수 6개

3. Homography Tr (Projective Tr) --> 미지수 8개


<img width=55% img src="https://github.com/user-attachments/assets/1d815074-8a72-47c9-a6c2-98cc95eeee8d">

위처럼 특징점을 기반으로 Transformation (== R|t) 를 찾아내야한다


<br/>

<img width=55% img src="https://github.com/user-attachments/assets/65d9b9f0-6cc8-49be-97fd-c201f841d063">

각각의 Tr마다 성질이 다름 

뉴클리디안은 똑같이 사각형, Affine은 평형사변형 가능, Homograhpy는 무작위 사각형으로 변형 가능하다

뉴클리디안은 스케일 변화도 X


### How de we get the transformation

그렇다면 어떻게 Tr을 얻을까?? --> 앞에 PPM matrix구하는것과 유사함

--> 여기부터 내일






