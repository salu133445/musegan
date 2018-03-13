function f_featureNorm = normalizeFeature(f_feature,normP, threshold)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Name: normalizeFeature
% Date of Revision: 2011-03
% Programmer: Meinard Mueller, Sebastian Ewert
%
% Description:
% - Normalizes a feature sequence according to the l^p norm
% - If the norm falls below threshold for a feature vector, then the
% normalized feature vector is set to be the unit vector.
%
% Input:
%         f_feature
%         normP
%         threshold
%
% Output:
%         f_featureNorm
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

f_featureNorm = zeros(size(f_feature));

% normalise the vectors according to the l^p norm
unit_vec = ones(1,12);
unit_vec = unit_vec/norm(unit_vec,normP);
for k=1:size(f_feature,2);
    n = norm(f_feature(:,k),normP);
    if n < threshold
        f_featureNorm(:,k) = unit_vec;
    else
        f_featureNorm(:,k) = f_feature(:,k)/n;
    end
end

end