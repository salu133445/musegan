function [ output_args ] = write_results(filename, result, label)
    if(nargin < 3) 
        lab = 0;
    else
        lab = 1;
    end;
    fid = fopen(filename, 'w');
    for i =1:length(result)-1
        if(lab)
            fprintf(fid,'%f %f %s\n' ,result(i), result(i+1), label(i));
        else
            fprintf(fid,'%f %f %s\n' ,result(i), result(i+1), 'unknown');
        end
    end
    fclose(fid);
end

