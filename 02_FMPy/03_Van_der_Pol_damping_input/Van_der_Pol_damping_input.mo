model Van_der_Pol_damping_input
  Real u(start = -1.0,fixed=true);
  Real v(start = 0.0,fixed=true);
  input Real damping(start=0.0);
  import Modelica.Math.sin;
equation
  der(u) = v;
  der(v) = damping - u + 1.2*sin(0.624*time);
end Van_der_Pol_damping_input;
