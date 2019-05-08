directoryname = 'E:\专业课程\人工智能\机器学习\code\降维\yalefaces';

files = dir( directoryname );
m = size( files, 1 );
G = zeros(243*320,m);

name =strcat(directoryname,'\');
temp = strcat( name , 'subject01.gif' );
c = imread( temp );
%imshow( c )
fprintf( files(1).name )

for i = 4 : m
    s = strcat( name, files(i).name );
    fig = imread( s );
    G(:,i-2) = reshape( fig, 243*320,1 );
end

k = 20;
%进行PCA分析
coeff=pca(G);
%只保留前k个特征
coeff(:,k+1:165)=0;

G = G* (coeff * coeff');

imshow( coeff );
%输出图片
filename = strcat( directoryname, '\' );
for i=1:166
    img=reshape(G(:,i),243,320);
    filename_m = strcat( filename, int2str(i),'.gif' );
    imwrite(img,filename_m);
    %saveas( img, filename_m, 'gif');
end
