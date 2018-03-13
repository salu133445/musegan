function visualizeCRP(f_crp,parameter)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Name: visualizeCRP
% Date of Revision: 2011-03
% Programmer: Meinard Mueller, Sebastian Ewert
%
% Description:
% Visualization of CRP features (wrapper around visualize_chroma)
%
% Input: 
%        f_CRP
%        parameter: is forwarded to  visualizeChroma
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
if isfield(parameter,'imagerange')==0
   parameter.imagerange = [-1 1];
end
if isfield(parameter,'colormap')==0
    blueSepia = gray(64).^0.8;
    blueSepia(:,1:2) = blueSepia(:,1:2) * 0.5;
    parameter.colormap = [flipud(blueSepia) ; hot(64)];
end

visualizeChroma(f_crp,parameter);

end