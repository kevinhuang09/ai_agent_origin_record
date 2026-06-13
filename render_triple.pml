load /home/tempadmin/agentic_ai/temp_ProtAnalyze.pdb, myprot
bg_color white
hide all
show cartoon, myprot
color spectrum, myprot
png /home/tempadmin/agentic_ai/render_ProtAnalyze_1_ribbon.png, width=800, height=600, ray=1
hide all
show sticks, myprot
color cpk, myprot
png /home/tempadmin/agentic_ai/render_ProtAnalyze_2_atoms.png, width=800, height=600, ray=1
hide all
show surface, myprot
set_color hydrophob, [1.0, 0.5, 0.5]
color hydrophob, resn ALA+VAL+LEU+ILE+PHE+TRP+MET+PRO
color slate, resn ASN+GLN+SER+THR+TYR+CYS
color marine, resn ARG+LYS+HIS+ASP+GLU
png /home/tempadmin/agentic_ai/render_ProtAnalyze_3_surface.png, width=800, height=600, ray=1
quit
