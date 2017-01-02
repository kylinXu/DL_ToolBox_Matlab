%% Preliminary codes for test PCA transformation 
%% by kylin

% 1.伸缩
% clf;
% A = [0, 1, 1, 0, 0;...
%     1, 1, 0, 0, 1];  % 原空间
% B = [3 0; 0 2];      % 线性变换矩阵
% 
% plot(A(1,:),A(2,:), '-*');hold on
% grid on;axis([0 3 0 3]); gtext('变换前');
% 
% Y = B * A;
% 
% plot(Y(1,:),Y(2,:), '-r*');
% grid on;axis([0 3 0 3]); gtext('变换后');

% 2. 切变
% clf;
% A = [0, 1, 1, 0, 0;...
%      1, 1, 0, 0, 1];   % 原空间
% B1 = [1 0; 1 1];       % 线性变换矩阵
% B2 = [1 0; -1 1];      % 线性变换矩阵
% B3 = [1 1; 0 1];       % 线性变换矩阵
% B4 = [1 -1; 0 1];      % 线性变换矩阵
% 
% Y1 = B1 * A;
% Y2 = B2 * A;
% Y3 = B3 * A;
% Y4 = B4 * A;
% 
% subplot(2,2,1);
% plot(A(1,:),A(2,:), '-*'); hold on;plot(Y1(1,:),Y1(2,:), '-r*');
% grid on;axis([-1 3 -1 3]);
% subplot(2,2,2);
% plot(A(1,:),A(2,:), '-*'); hold on;plot(Y2(1,:),Y2(2,:), '-r*');
% grid on;axis([-1 3 -1 3]);
% subplot(2,2,3);
% plot(A(1,:),A(2,:), '-*'); hold on;plot(Y3(1,:),Y3(2,:), '-r*');
% grid on;axis([-1 3 -1 3]);
% subplot(2,2,4);
% plot(A(1,:),A(2,:), '-*'); hold on;plot(Y4(1,:),Y4(2,:), '-r*');
% grid on;axis([-1 3 -1 3]);

% 3. 旋转

% 所有的变换其实都可以通过上面的伸缩和切变变换的到，如果合理地对变换矩阵B取值，能得到图形旋转的效果，如下，

clf;
A = [0, 1, 1, 0, 0;...
     1, 1, 0, 0, 1];  % 原空间
theta = pi/6;
B = [cos(theta) sin(theta); -sin(theta) cos(theta)];
Y = B * A;
figure;
plot(A(1,:),A(2,:), '-*'); hold on;plot(Y(1,:),Y(2,:), '-r*');
grid on;axis([-1 3 -1 3]);
