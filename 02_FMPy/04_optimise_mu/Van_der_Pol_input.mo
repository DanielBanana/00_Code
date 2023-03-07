model Van_der_Pol_input
  Real u(start = -1.0,fixed=true);
  Real v(start = 0.0,fixed=true);
  input Real mu(start=1.0);
  import Modelica.Math.sin;
equation
  der(u) = v;
  der(v) = mu*(1 - u*u)*v - u + 1.2*sin(0.624*time);
end Van_der_Pol_input;
