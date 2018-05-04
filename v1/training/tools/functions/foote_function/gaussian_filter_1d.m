function nc_m = gaussian_filter_1d(nc, winLen, variance)

    stdev = (variance* (winLen) * 0.5);
    gaussFilter = gausswin(winLen, (winLen-1)/2/stdev);
    gaussFilter = gaussFilter / sum(gaussFilter);   
    temp = conv(nc, gaussFilter);
    nc_m = temp(winLen/2:end-winLen/2);

end

