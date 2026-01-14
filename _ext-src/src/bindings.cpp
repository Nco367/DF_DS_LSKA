#include "/media/q/SSD2T/1linux/ku_github/PVN3D-master/pvn3d/_ext-src/include/ball_query.h"
#include "/media/q/SSD2T/1linux/ku_github/PVN3D-master/pvn3d/_ext-src/include/group_points.h"
#include "/media/q/SSD2T/1linux/ku_github/PVN3D-master/pvn3d/_ext-src/include/interpolate.h"
#include "/media/q/SSD2T/1linux/ku_github/PVN3D-master/pvn3d/_ext-src/include/sampling.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gather_points", &gather_points);
  m.def("gather_points_grad", &gather_points_grad);
  m.def("furthest_point_sampling", &furthest_point_sampling);

  m.def("three_nn", &three_nn);
  m.def("three_interpolate", &three_interpolate);
  m.def("three_interpolate_grad", &three_interpolate_grad);

  m.def("ball_query", &ball_query);

  m.def("group_points", &group_points);
  m.def("group_points_grad", &group_points_grad);
}
