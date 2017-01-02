%% Preliminary codes for test PCA transformation 
%% by kylin

% 1.����
% clf;
% A = [0, 1, 1, 0, 0;...
%     1, 1, 0, 0, 1];  % ԭ�ռ�
% B = [3 0; 0 2];      % ���Ա任����
% 
% plot(A(1,:),A(2,:), '-*');hold on
% grid on;axis([0 3 0 3]); gtext('�任ǰ');
% 
% Y = B * A;
% 
% plot(Y(1,:),Y(2,:), '-r*');
% grid on;axis([0 3 0 3]); gtext('�任��');

% 2. �б�
% clf;
% A = [0, 1, 1, 0, 0;...
%      1, 1, 0, 0, 1];   % ԭ�ռ�
% B1 = [1 0; 1 1];       % ���Ա任����
% B2 = [1 0; -1 1];      % ���Ա任����
% B3 = [1 1; 0 1];       % ���Ա任����
% B4 = [1 -1; 0 1];      % ���Ա任����
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

% 3. ��ת

% ���еı任��ʵ������ͨ��������������б�任�ĵ����������ضԱ任����Bȡֵ���ܵõ�ͼ����ת��Ч�������£�

clf;
A = [0, 1, 1, 0, 0;...
     1, 1, 0, 0, 1];  % ԭ�ռ�
theta = pi/6;
B = [cos(theta) sin(theta); -sin(theta) cos(theta)];
Y = B * A;
figure;
plot(A(1,:),A(2,:), '-*'); hold on;plot(Y(1,:),Y(2,:), '-r*');
grid on;axis([-1 3 -1 3]);
