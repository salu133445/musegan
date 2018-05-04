function P = kernel_density_estimation_2d(L, winLen_x, winLen_y, variance)
    stdev_t = (variance* (winLen_x) * 0.5);
    stdev_l = (variance* (winLen_y) * 0.5);
    gt = gausswin(winLen_x, (winLen_x-1)/2/stdev_t);
    gl = gausswin(winLen_y, (winLen_y-1)/2/stdev_l);
    G = (gt*gl');
    P_temp = conv2(L, G);
    conv_padding_x = abs((size(L,1) - size(P_temp,1))/2);
    conv_padding_y = abs((size(L,2) - size(P_temp,2))/2);
    P = P_temp(1+conv_padding_x:end-conv_padding_x,1+conv_padding_y:end-conv_padding_y);
end

