function visualizePitch(f_pitch,parameter)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Name: visualizePitch
% Date of Revision: 2011-03
% Programmer: Meinard Mueller, Sebastian Ewert
%
% Description:
% Visualization of f_pitch
%
% Input:
%        f_pitch
%        parameter.midiMin = 21;
%        parameter.midiMax = 108;
%        parameter.featureRate = 0;
%        parameter.colorbar = 1;
%        parameter.colormap = hot2;
%        parameter.print = 0;
%        parameter.printFile = 'figure.eps';
%        parameter.printDir = '';
%        parameter.title = '';
%        parameter.xlabel = '';
%        parameter.ylabel = '';
%        parameter.imagerange = 0; %[0 1];    % 0 means automatic
%        parameter.usePitchNameLabels = 0;
%        parameter.PitchNameLabels = ...
%        parameter.Ytick = [1 10 20 30 40 50 60 70 80 90 100 110 120]; % not used when usePitchNameLabels==1
%        parameter.printPaperPosition = [1   10   26  15]; %[left, bottom, width, height]
%        parameter.fontSize = 0;  % 0 means automatic
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

if isfield(parameter,'midiMin')==0
    parameter.midiMin = 21;
end
if isfield(parameter,'midiMax')==0
    parameter.midiMax = 108;
end
if isfield(parameter,'featureRate')==0
    parameter.featureRate = 0;
end
if isfield(parameter,'colorbar')==0
    parameter.colorbar = 1;
end
if (isfield(parameter,'colormap')==0) || (isstr(parameter.colormap) && strcmpi(parameter.colormap,'hot2'))
    hot2 = hot(64);
    hot2 = [hot2; hot2(32:64,:); hot2(32:64,:)];
    hot2 = sort(hot2);
    parameter.colormap = hot2;
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
if isfield(parameter,'ylabel')==0
    parameter.ylabel = '';
end
if isfield(parameter,'imagerange')==0
    parameter.imagerange = 0; %[0 1];
end
if isfield(parameter,'usePitchNameLabels')==0
    parameter.usePitchNameLabels = 0;
end
if isfield(parameter,'PitchNameLabels')==0
    parameter.PitchNameLabels = ...
        ['   ';'   ';'   ';'   ';'   ';'   ';'   ';'   ';'   ';'   ';'   ';...
        'C0 ';'   ';'   ';'   ';'   ';'   ';'   ';'   ';'   ';'   ';'   ';'   ';...
        'C1 ';'   ';'   ';'   ';'   ';'   ';'   ';'   ';'   ';'   ';'   ';'   ';...
        'C2 ';'   ';'   ';'   ';'   ';'   ';'   ';'   ';'   ';'   ';'   ';'   ';...
        'C3 ';'   ';'   ';'   ';'   ';'   ';'   ';'   ';'   ';'   ';'   ';'   ';...
        'C4 ';'   ';'   ';'   ';'   ';'   ';'   ';'   ';'   ';'   ';'   ';'   ';...
        'C5 ';'   ';'   ';'   ';'   ';'   ';'   ';'   ';'   ';'   ';'   ';'   ';...
        'C6 ';'   ';'   ';'   ';'   ';'   ';'   ';'   ';'   ';'   ';'   ';'   ';...
        'C7 ';'   ';'   ';'   ';'   ';'   ';'   ';'   ';'   ';'   ';'   ';'   ';...
        'C8 ';'   ';'   ';'   ';'   ';'   ';'   ';'   ';'   ';'   ';'   ';'   ';...
        'C9 '];
end
if isfield(parameter,'Ytick')==0
    parameter.Ytick = [1 10 20 30 40 50 60 70 80 90 100 110 120]; % not used when usePitchNameLabels==1
end
if isfield(parameter,'printPaperPosition')==0
    parameter.printPaperPosition = [1   10   26  15]; %[left, bottom, width, height]
end
if isfield(parameter,'fontSize')==0
    % 0 means automatic
    parameter.fontSize = 0;
end
if isfield(parameter,'createAxisLabel')==0
    parameter.createAxisLabel = 1;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Visualization
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure;

if parameter.featureRate == 0
    t_axis = (1:size(f_pitch,2));
else
    t_axis = (0:size(f_pitch,2)-1)/parameter.featureRate;
end

p_axis = (parameter.midiMin:parameter.midiMax);
f_image = f_pitch(p_axis,:);

if parameter.imagerange == 0
    imagesc(t_axis,p_axis,f_image);
else
    imagesc(t_axis,p_axis,f_image,parameter.imagerange);
end

set(gca,'YTick',parameter.Ytick);
set(gca,'YDir','normal');

if parameter.usePitchNameLabels
    set(gca,'YTick',[parameter.midiMin:parameter.midiMax]);
    set(gca,'YTickLabel',parameter.PitchNameLabels(parameter.midiMin:parameter.midiMax,:));
end

title(parameter.title);
xlabel(parameter.xlabel);
ylabel(parameter.ylabel);

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


