function out = result_spatial(phi, geometry, figureDir, prefix)
%RESULT_SPATIAL 在给定几何下展示各阶振型的空间形状。
%
%   OUT = bayoma.result_spatial(PHI, GEOMETRY, FIGUREDIR) 逐阶绘制振型的三维形状
%   并保存为 PNG，返回结构体字段 available、reason、channel_names、connections、
%   figures。GEOMETRY 为描述文件里的几何段，经 bayoma.result_geometry 规范化。
%
%   OUT = bayoma.result_spatial(..., PREFIX) 指定文件名前缀，缺省 'mode'。
%
%   未给 FIGUREDIR 时只返回可用于绘图的数据：available 为 false，reason 说明原因，
%   figures 为空。缺省几何时同样不绘图。
%
%   图中未变形位置用灰色虚线连成结构简图，变形后位置用实线连接并以散点标出测点；
%   变形量按结构尺寸的 1/4 缩放，复振型分两栏给出实部与虚部。

if nargin < 4 || isempty(prefix)
    prefix = 'mode';
end

nch = size(phi, 1);
m = size(phi, 2);
geo = bayoma.result_geometry(geometry, nch, 0:nch - 1);

out = struct();
out.available = false;
out.reason = bayoma.json_null();
names = cell(1, nch);
for i = 1:nch
    names{i} = geo.channels{i}.name;
end
out.channel_names = names;
out.connections = geo.connections;
out.figures = {};

if isempty(figureDir)
    out.reason = '未指定 figure_dir，仅返回几何与振型数据';
    return;
end

if ~exist(figureDir, 'dir')
    mkdir(figureDir);
end

figs = cell(1, m);
for i = 1:m
    path = fullfile(figureDir, sprintf('%s_%02d.png', prefix, i - 1));
    plot_mode_shape(sprintf('阶 %d', i), phi(:, i), geo.positions, ...
        geo.directions, geo.connections, path);
    rec = struct();
    rec.order = i - 1;
    rec.path = path;
    rec.complex = logical(~isreal(phi(:, i)));
    figs{i} = rec;
end
out.figures = figs;
out.available = true;
end

% ----------------------------------------------------------------------
function plot_mode_shape(name, shape, positions, directions, connections, outPath)
%PLOT_MODE_SHAPE 画一阶振型的空间形状（未变形虚线 + 变形实线 + 测点散点）。
%   复振型分两栏：左栏实部、右栏虚部；实振型只画一栏。
cplx = ~isreal(shape);
if cplx
    panels = 2;
else
    panels = 1;
end
fig = figure('Visible', 'off', 'Position', [100, 100, 520 * panels, 500]);
amp = max(abs(shape));
if amp <= 0
    amp = 1;
end
span = max(max(positions) - min(positions));
if span <= 0
    span = 1;
end
scale = 0.25 * span / amp;

for p = 1:panels
    subplot(1, panels, p);
    if ~cplx || p == 1
        comp = real(shape);
    else
        comp = imag(shape);
    end
    if cplx && p == 1
        label = '实部';
    elseif cplx
        label = '虚部';
    else
        label = '幅值';
    end
    hold on;
    % 未变形位置：灰色虚线，给出结构简图
    for k = 1:numel(connections)
        pair = connections{k};
        a = pair(1) + 1;
        b = pair(2) + 1;
        plot3(positions([a, b], 1), positions([a, b], 2), positions([a, b], 3), ...
            '--', 'Color', [0.75, 0.75, 0.75], 'LineWidth', 0.9);
    end
    % 变形后位置：沿测点方向平移，形成为振型形状
    moved = positions + scale * (comp * ones(1, 3)) .* directions;
    for k = 1:numel(connections)
        pair = connections{k};
        a = pair(1) + 1;
        b = pair(2) + 1;
        plot3(moved([a, b], 1), moved([a, b], 2), moved([a, b], 3), ...
            '-', 'Color', [0, 0.447, 0.741], 'LineWidth', 1.6);
    end
    scatter3(moved(:, 1), moved(:, 2), moved(:, 3), 34, comp, 'filled');
    hold off;
    grid on;
    box on;
    view(3);
    title(sprintf('%s · %s（%s，缩放 %.3g）', name, 'mode', label, scale), ...
        'FontSize', 9);
    xlabel('x');
    ylabel('y');
    zlabel('z');
    set(gca, 'FontSize', 7);
end
% 图形在无显示环境下生成，按 150 dpi 写出位图后关闭
print(fig, outPath, '-dpng', '-r150');
close(fig);
end
