model simple_example
  Real x(start = 0.5,fixed=true);
equation
  der(x) = 10*x;
end simple_example;
