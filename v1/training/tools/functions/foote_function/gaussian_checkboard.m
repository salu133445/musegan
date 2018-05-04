function G= gaussian_checkboard(winLen, variance)
    stdev = (variance* (winLen) * 0.5);
    g = gausswin(winLen, (winLen-1)/2/stdev);
    G = (g*g');
    G(winLen/2:end, 1:winLen/2) = -1* G(winLen/2:end, 1:winLen/2) ;
    G(1:winLen/2, winLen/2:end) = -1* G(1:winLen/2, winLen/2:end);
end

