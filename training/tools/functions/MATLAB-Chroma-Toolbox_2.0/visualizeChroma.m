function visualizeChroma(f_chroma,parameter)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Name: visualizeChroma
% Date of Revision: 2011-03
% Programmer: Meinard Mueller, Sebastian Ewert
%
% Description:
% Visualization of f_chroma
%
% Input:
%        f_chroma
%        parameter.featureRate = 0;  % 0 means unknown
%        parameter.colorbar = 1;
%        parameter.colormap = 'hot';
%        parameter.print = 0;
%        parameter.printFile = 'figure.eps';
%        parameter.printDir = '';
%        parameter.title = '';
%        parameter.xlabel = '';
%        parameter.imagerange = [0 1];   % 0 means automatic
%        parameter.fontSize = 0;   % 0 means automatic
%        parameter.printPaperPosition = [1   10   26  15]; %[left, bottom, width, height]
%        parameter.createAxisLabel = 1;
%
% License:
%     This file is part of 'Chroma Toolbox'.
%
%     'Chroma Toolbox' is free software: you can redistribute it and/or modify
%     it under the terms of the GNU General Public License as published by
%     the Free Software Foundation, either version 2 of the License, or
%     (at your option) any later version.
%
%     'Chroma Toolbox' is distributed in the hope that it will be useful,
%     but WITHOUT ANY WARRANTY; without even the implied warranty of
%     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%     GNU General Public License for more details.
%
%     You should have received a copy of the GNU General Public License
%     along with 'Chroma Toolbox'. If not, see <http://www.gnu.org/licenses/>.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nargin<2
    parameter=[];
end

if isfield(parameter,'featureRate')==0
    parameter.featureRate = 0;
end
if isfield(parameter,'colorbar')==0
    parameter.colorbar = 1;
end
if isfield(parameter,'colormap')==0
    parameter.colormap = 'hot';
end
if isfield(parameter,'print')==0
    parameter.print = 0;
end
if isfield(parameter,'printFile')==0
    parameter.printFile = 'figure.eps';
end
if isfield(parameter,'printDir')==0
    parameter.printDir = '';
end
if isfield(parameter,'title')==0
    parameter.title = '';
end
if isfield(parameter,'xlabel')==0
    parameter.xlabel = '';
end
if isfield(parameter,'imagerange')==0
    parameter.imagerange = [0 1];
end
if isfield(parameter,'fontSize')==0
    % 0 means automatic
    parameter.fontSize = 0;
end
if isfield(parameter,'printPaperPosition')==0
    parameter.printPaperPosition = [1   10   26  15]; %[left, bottom, width, height]
end
if isfield(parameter,'createAxisLabel')==0
    parameter.createAxisLabel = 1;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Visualization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

seg_num = size(f_chroma,2);

chroma_names                 = ['C ';'C#';'D ';'D#';'E ';'F ';'F#';'G ';'G#';'A ';'A#';'B '];
figure;
set(gcf,'renderer','painters');

if parameter.featureRate == 0
    t = (1:seg_num);
else
    t = (0:seg_num-1)/parameter.featureRate;
end

if all(parameter.imagerange == 0)
    imagesc(t,[1:12],f_chroma);
else
    imagesc(t,[1:12],f_chroma,parameter.imagerange);
end
set(gca,'YTick',[1:12]);
set(gca,'YTickLabel',chroma_names);
set(gca,'YDir','normal');
if t(end)>t(1)
    set(gca,'XLim',[t(1),t(end)]);
end

title(parameter.title);
xlabel(parameter.xlabel);

if ~parameter.createAxisLabel
    set(gca, 'XTick', [], 'YTick', [])
end

colormap(parameter.colormap);

if parameter.fontSize
    set(gca,'FontSize',parameter.fontSize)
end

if parameter.colorbar == 1
    hColorbar = colorbar;
    if parameter.fontSize
        set(hColorbar,'FontSize',parameter.fontSize)
    end
end
drawnow;

if parameter.print == 1
    set(gcf,'PaperPosition',parameter.printPaperPosition);
    print('-depsc2',strcat(parameter.printDir,parameter.printFile));
end

end


