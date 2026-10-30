from beam_analysis import Beam, BeamType, plot_diagrams

def komplex_pelda(show_plots=True):
    print("=" * 60)
    print(" GERENDA STATIKAI ÉS SZILÁRDSÁGTANI ELEMZÉS")
    print("=" * 60)
    
    beam = Beam(length=10.0, beam_type=BeamType.SIMPLY_SUPPORTED)
    
    beam.add_point_load(force=15e3, x=3.0)
    beam.add_point_load(force=20e3, x=7.0)
    beam.add_distributed_load(q_start=5.0e3, q_end=8.0e3, x_start=4.0, x_end=9.0)
    
    if hasattr(beam, 'set_cross_section'):
        beam.set_cross_section(shape="I", height=0.2, width=0.1, allowable_stress=160e6)

    result = beam.solve()

    if hasattr(result, 'print_summary'):
        result.print_summary()
    else:
        print(f"Típus: kéttámaszú gerenda")
        print(f"Hossz: L = {beam.length:.2f} m")
        print(f"Támaszok: x = 0.000 m és x = {beam.length:.2f} m")
        
        print("\n[Támaszreakciók]:")
        for r in result.reactions:
            print(f"  {r.name} = {r.value / 1e3:.3f} kN (hely: x = {r.x} m)")
            
        print(f"\n[Igénybevételi szélsőértékek]:")
        if hasattr(result, 'max_moment'):
            print(f"  M_max = {result.max_moment.value / 1e3:.2f} kNm (x = {result.max_moment.x:.3f} m)")
        if hasattr(result, 'max_shear'):
            print(f"  T_max = {result.max_shear.value / 1e3:.2f} kN (x = {result.max_shear.x:.3f} m)")

    print("=" * 60)

    if show_plots:
        plot_diagrams(result)

if __name__ == "__main__":
    komplex_pelda(show_plots=True)