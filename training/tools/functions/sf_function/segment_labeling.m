function [label, Stran ]= segment_labeling( b, R, vis)
 
    ss = length(b);
    num_s = ss-1;
    Rs = cell(num_s);
    S = zeros(num_s);
    for u = 1:num_s
        for v = 1:ss-1
            Rs{u, v} = R(b(u):b(u+1),b(v):b(v+1) );
        end
    end

    for u = 1:num_s
        for v = 1:num_s
            r_temp = Rs{u, v};
            [lu, lv] = size(r_temp);
            q_temp = zeros(lu, lv);

            for i = 1: lu
                for j =1: lv

                    if(i-1<1 || j-1 <1) 
                        cand1 = 0;
                    else
                        cand1 = q_temp(i-1,j-1);
                    end


                    if(i-2<1 || j-1 <1) 
                        cand2 = 0;
                    else
                        cand2 = q_temp(i-2,j-1);
                    end

                    if(i-1<1 || j-2 <1) 
                        cand3 = 0;
                    else
                        cand3 = q_temp(i-1,j-2);
                    end

                    q_temp(i, j) = max([cand1, cand2, cand3]) + r_temp(i, j);

                end
            end

            S(u, v) =  max(max(q_temp)) / min([lu, lv]);


        end
    end
    S_unthres = S;
    thr = std(reshape(S, [],1)) + mean(reshape(S, [],1));
    for u = 1:num_s
        for v = 1:num_s
            if(S(u,v) < thr)
                S(u, v) = 0;
            end
        end
    end

    if(vis)
        figure;
        colormapSet = generateColormapValue();
        colormap(colormapSet.colormap5);
        imagesc(S_unthres);
         figure;
        colormapSet = generateColormapValue();
        colormap(colormapSet.colormap5);
        imagesc(S);
    end

    Stran = S;
    iter_limit = 1000;
    iter = 1;
    while(all(all(Stran==S==1)==1))
        Stran = Stran *S;  
        iter = iter +1;
        if(iter > iter_limit) break; end
    end
    for u = 1:num_s
        for v = 1:num_s
            if(Stran(u,v) >= 1)
                Stran(u, v) = 1;
            else
                Stran(u, v) = 0;
            end
        end
    end
    if(vis)
        figure;
        colormapSet = generateColormapValue();
        colormap(colormapSet.colormap5);
        imagesc(Stran);
    end

    label = zeros(1, num_s);
    label_index = 65;
    for i =1:num_s
        if(label(i) == 0)
            label(i) = label_index;

            for j = i+1:num_s
                if(Stran(i,j) == 1 && label(j) == 0)
                    label(j) = label_index;
                end
            end
            label_index = label_index +1;
        else
            continue;
        end
    end


end

