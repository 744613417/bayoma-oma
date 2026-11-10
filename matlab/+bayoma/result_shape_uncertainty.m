function items = result_shape_uncertainty(covPhi, phi, channelIndex)
%RESULT_SHAPE_UNCERTAINTY 各阶振型的识别不确定性。
%
%   ITEMS = bayoma.result_shape_uncertainty(COVPHI, PHI, CHANNELINDEX) 返回元胞
%   数组，每项一阶模态，字段为 order、complex、coefv、channel_index、
%   std_per_channel、ci95_halfwidth_per_channel、shape、cov。
%
%   COVPHI 为元胞，每项是该阶振型在通道空间的协方差矩阵；实振型为 n x n，
%   复振型按「实部在前、虚部在后」排布为 2n x 2n。PHI 为振型矩阵，每列一阶。
%
%   变异系数取通道协方差矩阵迹的平方根：单位范数振型的扰动能量用迹度量，
%   与振型不确定度在切空间内的定义一致。复振型取实部与虚部两块对角之和，
%   即各通道扰动的模方差。95% 区间半宽按标准差的 1.96 倍给出。

if nargin < 3 || isempty(channelIndex)
    channelIndex = 0:size(phi, 1) - 1;
end

m = size(phi, 2);
nch = size(phi, 1);
items = cell(1, m);
for i = 1:m
    shape = phi(:, i);
    cplx = ~isreal(shape);
    if i <= numel(covPhi) && ~isempty(covPhi{i})
        cov = double(covPhi{i});
    else
        cov = zeros(nch);
    end
    if cplx
        n = numel(shape);
        if size(cov, 1) == 2 * n
            covRe = cov(1:n, 1:n);
            covIm = cov(n + 1:end, n + 1:end);
        else
            covRe = zeros(n);
            covIm = zeros(n);
        end
        varCh = max(diag(covRe), 0) + max(diag(covIm), 0);
    else
        varCh = max(diag(cov), 0);
    end
    tr = sum(varCh);
    stdCh = sqrt(varCh);
    rec = struct();
    rec.order = i - 1;
    rec.complex = logical(cplx);
    if tr > 0
        rec.coefv = sqrt(tr);
    else
        rec.coefv = 0;
    end
    rec.channel_index = double(channelIndex(:).');
    rec.std_per_channel = stdCh(:).';
    rec.ci95_halfwidth_per_channel = 1.96 * stdCh(:).';
    % 复振型按 re/im 分开给出，与结果文件里复数的写法一致
    if cplx
        rec.shape = struct('re', real(shape(:).'), 'im', imag(shape(:).'));
    else
        rec.shape = double(shape(:).');
    end
    rec.cov = cov;
    items{i} = rec;
end
end
