% This file includes the segmentation part
%In the first part it is going convert the RGB image to the gray scale image 
function [Img_seg, ai, ci, col, dia] = GUI_Seg(I)

I_G = rgb2gray(I);
I_G = im2double(I_G);
% Fft and Butterworth’s High Pass Filter to filter out the gravity component from the acceleration data. 
Fft_I = fft2(log(I_G + 0.01));
Fil_I = Butterworth_HighPassFilter(I_G, 500, 3);
temp = Fft_I .* Fil_I;
h = real(ifft2(temp));
h = exp(h);
Img = ifftshow(h);
%divide image into segments and store the values separetly, we can see below as red, green and blue  
[Img] = segment(Img);

I_R = I(:,:,1);
I_G = I(:,:,2);
I_B = I(:,:,3);
%Convert double type of image to uint8 for fastest histogram computation
% uint8 is used unsigned 8 bit integer. And that is the range of pixel
Img_seg = uint8(Img);
%use metrix multiplication, After that the values pass to the cat function 
I_R = I_R .* Img_seg;
I_G = I_G .* Img_seg;
I_B = I_B .* Img_seg;

Img_seg = cat(3, I_R, I_G, I_B);
%create the new file(called jpg file) in current folder
imwrite(Img_seg, 'x.jpg');
%features going to extracted using ABCD rule
[ai, ci, dia] = assym_bord_dia(Img);

dia = nor_dia(dia);

col = colour(I, Img);

end
%used Otsu's thresholding algorithm, it helps to differentiates between the background and the foreground of the image
function [t] = otsu_impl(counts)

num_bins = numel(counts);

counts = double( counts(:) );

p = counts / sum(counts);
omega = cumsum(p);
mu = cumsum(p .* (1:num_bins)');
mu_t = mu(end);

sigma_b_squared = (mu_t * omega - mu).^2 ./ (omega .* (1 - omega));
% Find the location of the maximum value of sigma_b_squared.
% The maximum may extend over several bins, so average together the locations.  If maxval is NaN, meaning that sigma_b_squared is all NaN, then return 0

maxval = max(sigma_b_squared);
isfinite_maxval = isfinite(maxval);
if isfinite_maxval
    idx = mean(find(sigma_b_squared == maxval));
    t = (idx - 1) / (num_bins - 1);
else
    t = 0.0;
end

end

function [level] = thresh_impl(I)

if ~isempty(I)
    
    I = im2uint8(I(:));
    num_bins = 256;
    counts = imhist(I,num_bins);
    
    level = otsu_impl(counts);
    
else
    level = 0.0;
end
end
%Butterworth Highpass Filter is used and it will do image sharpening in the frequency domain
function [out] = Butterworth_HighPassFilter(I, d, n)
h = size(I, 1);
w = size(I, 2);
[x, y] = meshgrid(-floor(w / 2):floor(w - 1) / 2, -floor(h / 2):floor(h - 1) / 2);
out = 1 ./ (1 + (d ./ (x .^ 2 + y .^ 2) .^ 0.5) .^ (2 * n));
end

function [I] = ifftshow(f)
fabs = abs(f);
fmax = max(fabs(:));
I = (fabs / fmax);
end

function [outImage] = segment(I)
% Saving the size of the input_image in pixels-
% M : no of rows (height of the image)
% N : no of columns (width of the image)
%[M, N] = size(input_image

[m, n] = size(I);

level = thresh_impl(I);     
%imbinarize(I) creates a binary image from 2-D or 3-D grayscale image 
I_Bin = imbinarize(I,level);   

K = medfilt2(I_Bin);
se = strel('disk', 4);

hairs = imbothat(K,se);      
woHair = K;

for i = 1:m
    for j =  1:n
        if hairs(i, j) == 1
            woHair(i,j) = 1;
        end
    end
end

%produce eroded and noice removed image, as well as dilation also done


woHair = imerode(woHair, se);

woHair_Edge = edge(woHair,'Canny');

se90 = strel('line', 2, 90);
se0 = strel('line', 2, 0);

woHair_Edge_Dil = imdilate(woHair_Edge, [se90 se0]);
% morphological operations as hole filling and filtering
outImage = imfill(woHair_Edge_Dil, 'holes');

outImage = bwareafilt(outImage, 1);

end

function [ai, ci, d] = assym_bord_dia(Img)
    %Calculation centroids for connected components in the image using region props
    props = regionprops(Img, 'Centroid', 'Orientation');
    xCentroid = props.Centroid(1);
    yCentroid = props.Centroid(2);

    [rows, columns] = size(Img);
    
    middlex = round(columns/2);
    middley = round(rows/2);

    deltax = middlex - xCentroid;
    deltay = middley - yCentroid;
    I = imtranslate(Img,  [deltax, deltay]);

    angle = -props.Orientation;
    I = imrotate(I, angle, 'crop');
    
    temp_up = I;
    temp_left = I;
    
    for h = middley:rows
        for k = 1:columns
            temp_up(h, k) = 0;
        end
    end
    
    for k = middlex:columns
        for h = 1:rows
            temp_left(h, k) = 0;
        end
    end
    
    temp_down = I - temp_up;
    temp_right = I - temp_left;
    
    temp_up = flipud(temp_up);
    temp_left = fliplr(temp_left);
    
    temp_x = xor(temp_up, temp_down);
    for m = 1:columns
        temp_x(middley + 1, m) = 0;
        temp_x(middley, m) = 0;
    end
    
    temp_y = xor(temp_left, temp_right);
    for m = 1:rows
        temp_y(m, middlex + 1) = 0;
        temp_y(m, middlex) = 0;
    end
    %calculation of  Assymetry index 
    Ax = sum(temp_x(:));
    Ay = sum(temp_y(:));
    A = sum(I(:));
    
    ai = (Ax + Ay) / A;
    
    I_E = bwperim(I);
    edge_coord_x = [];
    edge_coord_y = [];
    
    k = 0;
    max_dist = 0;
    
    for i = 1:rows
        for j = 1:columns
            if I_E(i, j) == 1
                k = k + 1;
                edge_coord_x(k) = i;
                edge_coord_y(k) = j;
            end
        end     
    end
    %calculation of Border, here ci means compactness index
    ci = ((k .^ 2) ./ (4 .* pi .* A));
    
    for i = 1:k
        x = edge_coord_x(i);
        y = edge_coord_y(i);
        for j = 1:k
            dist = sqrt( (edge_coord_x(j) - x) .^ 2 + (edge_coord_y(j) - y) .^ 2 );
            if dist > max_dist
                max_dist = dist;
            end
        end
    end

    d = max_dist;
end
%color extraction
function [c] = colour(I, Img)

I_R = I(:,:,1);
I_G = I(:,:,2);
I_B = I(:,:,3);

[m ,n] = size(Img);
Img = uint8(Img);

I_R = I_R .* Img;
I_G = I_G .* Img;
I_B = I_B .* Img;

I_R = double(I_R) / 255;
I_G = double(I_G) / 255;
I_B = double(I_B) / 255;

white = [197 188 217];
black = [41 31 30];
red = [118 21 17];
light_brown = [163 82 16];
dark_brown = [135 44 5];
blue_grey = [113 108 139];

white = white / 255;
black = black / 255;
red = red / 255;
light_brown = light_brown / 255;
dark_brown = dark_brown / 255;
blue_grey = blue_grey / 255;

count_white = 0;
count_black = 0;
count_red = 0;
count_light_brown = 0;
count_dark_brown = 0;
count_blue_grey = 0;

for i = 1:m
    for j = 1:n
        if (Img(i, j) == 1)
            dis_white = sqrt ((I_R(i, j) - white(1)) ^ 2 + (I_G(i, j) - white(2)) ^ 2 + (I_B(i, j) - white(3)) ^ 2);
            dis_black = sqrt ((I_R(i, j) - black(1)) ^ 2 + (I_G(i, j) - black(2)) ^ 2 + (I_B(i, j) - black(3)) ^ 2);
            dis_red = sqrt ((I_R(i, j) - red(1)) ^ 2 + (I_G(i, j) - red(2)) ^ 2 + (I_B(i, j) - red(3)) ^ 2);
            dis_light_brown = sqrt ((I_R(i, j) - light_brown(1)) ^ 2 + (I_G(i, j) - light_brown(2)) ^ 2 + (I_B(i, j) - light_brown(3)) ^ 2);
            dis_dark_brown = sqrt ((I_R(i, j) - dark_brown(1)) ^ 2 + (I_G(i, j) - dark_brown(2)) ^ 2 + (I_B(i, j) - dark_brown(3)) ^ 2);
            dis_blue_grey = sqrt ((I_R(i, j) - blue_grey(1)) ^ 2 + (I_G(i, j) - blue_grey(2)) ^ 2 + (I_B(i, j) - blue_grey(3)) ^ 2);
            
            if dis_white < 0.35
                count_white = count_white + 1;
            end
            if dis_black < 0.35
                count_black = count_black + 1;
            end
            if dis_red < 0.35
                count_red = count_red + 1;
            end
            if dis_light_brown < 0.35
                count_light_brown = count_light_brown + 1;
            end
            if dis_dark_brown < 0.35
                count_dark_brown = count_dark_brown + 1;
            end
            if dis_blue_grey < 0.35
                count_blue_grey = count_blue_grey + 1;
            end
            
        end
    end
end

Img_pixel_count = nnz(Img);
thresh_count = 0.05 * Img_pixel_count;

colour_count = 0;

if count_white > thresh_count
    colour_count = colour_count + 1;
end
if count_black > thresh_count
    colour_count = colour_count + 1;
end
if count_red > thresh_count
    colour_count = colour_count + 1;
end
if count_light_brown > thresh_count
    colour_count = colour_count + 1;
end
if count_dark_brown > thresh_count
    colour_count = colour_count + 1;
end
if count_blue_grey > thresh_count
    colour_count = colour_count + 1;
end

c = colour_count;
end
%diameter  and only the points along the edge of the lesion are to be considered to find the diameter


function [nd] = nor_dia(d)
    nd = (((d - 0) / (600 - 0)) * (10 - 0)) + 0;
end