function print_fig(h,F)
% h: figure handle
% fname: name of fig file & script
% wh: set paper width and height
if ~isfield(F,'renderer'), F.renderer='painters'; end
if isfield(F,'wh')
    F.PaperSize=F.wh;
    F.PaperPosition=[0 0 F.wh];
end
if ~isfield(F,'png'), F.png=0; end
if ~isfield(F,'svg'), F.svg=0; end
if ~isfield(F,'fig'), F.fig=0; end
if ~isfield(F,'pdf'), F.pdf=1; end % default only plot pdf
if ~isfield(F,'PaperSize'), F.PaperSize=[2 2]; end
if ~isfield(F,'PaperPosition'), F.PaperPosition=[0 0 F.PaperSize]; end
if ~isfield(F,'fname'), F.fname='temp_fig'; end
if ~isfield(F,'PaperPositionMode'), F.PaperPositionMode='auto'; end
set(h,'PaperSize',F.PaperSize,'PaperPosition',F.PaperPosition,'color','w');
set(h, 'InvertHardCopy', 'off');
set(h,'renderer',F.renderer)
if F.pdf==1, print(h,F.fname,'-dpdf'), end
if F.svg==1, print(h,F.fname,'-dsvg'), end
if F.png==1, print(h,F.fname,'-dpng','-r300'), end
if F.fig==1, saveas(h,F.fname,'fig'), end