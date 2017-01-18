% H = SCALEBAR( POS, TXT)
%
% draws a scalebar in the current axis, POS is the position
% (in axis units) and TXT is the label.

function h = scalebar( position, txt );

h = axes( 'pos', position );
hold on;

if position(3)>position(4),   %horizontal scale bar
  plot( [0 1], [0.5 0.5], 'k-', 'LineWidth', 1 );
  plot( [0 0], [0 1], 'k-', 'LineWidth', 1);
  plot( [1 1], [0 1], 'k-', 'LineWidth', 1);
  axis([0 1 0 1]);
  axis off;
  text( 0.5, 0, txt, 'HorizontalAlignment', 'Center', 'VerticalAlignment', ...
        'Top');
else
  plot( [0.5 0.5], [0 1], 'k-', 'LineWidth', 1 );
  plot( [0 1], [0 0], 'k-', 'LineWidth', 1);
  plot( [0 1], [1 1], 'k-', 'LineWidth', 1);
  axis([0 1 0 1]);
  axis off;
  text( 1, 0.5, txt, 'HorizontalAlignment', 'Left', 'VerticalAlignment', ...
        'middle', 'Rotation', 90);
end  
