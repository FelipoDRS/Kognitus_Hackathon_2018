%Script for data cleaning, processing and fitting in Kognitus 2018
%Hackathon
%Group: Klepper
y=zeros(810,1);
x=zeros(810,7);
n=1;
for i=0:5%big column
for j=0:4%big line
for l=0:4%small column
for m=0:5%small line
x(n,1)=CasingWear1S1{1,3+8*i};
x(n,2)=CasingWear1S1{2,3+8*i};
x(n,3)=CasingWear1S1{3,3+8*i};
x(n,4)=CasingWear1S1{4,3+8*i};
x(n,5)=j;
if isnan(CasingWear1S1{11+11*j+m,8*i+1})==false
x(n,6)=CasingWear1S1{11+11*j+m,8*i+1};
x(n,7)=CasingWear1S1{9+11*j,8*i+l+2};
y(n)=CasingWear1S1{11+11*j+m,2+8*i+l};
n=n+1;
end
end
end
end
end

x = x'; %matlab accepts collumn vectors
y = y';

trainFcn = 'trainbr';  
hiddenLayerSize = 5;
net = fitnet(hiddenLayerSize,trainFcn);

net.input.processFcns = {'removeconstantrows','mapminmax'}; %preprocessing data
net.output.processFcns = {'removeconstantrows','mapminmax'};

net.divideFcn = 'dividerand';
net.divideMode = 'sample'; 
net.divideParam.trainRatio = 60/100; 
net.divideParam.valRatio = 20/100;
net.divideParam.testRatio = 20/100;%a lot to test
[net,tr] = train(net,x,y);
o = net(x);

[K,M]=meshgrid(0:35,0.01:0.01:0.5); %plotting the prediction

Y=zeros(size(K));
for i=1:size(K,1)
for j=1:size(K,2)
Y(i,j)=sim(net,[273.05,20.24,13.49,758.4235,0,K(i,j),M(i,j)]');
end
end
mesh(K,M,Y*100)%converting to percentage
xlabel('Wear (%)')
ylabel('Ovality(%)')
zlabel('Reduction factor(%)')
title('13.49 D/t ratio, full case')
