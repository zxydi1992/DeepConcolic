from nc_setup_badmodel import *
from nc_l0 import *

def run_nc_l0(test_object, outs):
  nc_results, layer_functions, cover_layers, activations, test_cases, adversarials=nc_setup(test_object, outs)
  d_advs=[]

  count = 0
  feasible_count = 0
  found_count = 0

  while True:
    count += 1

    index_nc_layer, nc_pos, nc_value=get_nc_next(cover_layers, test_object.layer_indices)
    nc_layer=cover_layers[index_nc_layer]

    shape=np.array(nc_layer.activations).shape
    pos=np.unravel_index(nc_pos, shape)
    im=test_cases[pos[0]]
    act_inst=eval(layer_functions, im)
    s=pos[0]*int(shape[1]*shape[2])
    if nc_layer.is_conv:
      s*=int(shape[3])*int(shape[4])

    feasible, d, new_im = l0_negate(test_object.test_dnn, layer_functions, [im], nc_layer, nc_pos-s)

    cover_layers[index_nc_layer].disable_by_pos(pos)
    d_adv=-1

    if feasible:

      if l0_filtered(test_object.raw_data.data, new_im): 
        continue
      feasible_count += 1

      test_cases.append(new_im)
      update_nc_map_via_inst(cover_layers, eval(layer_functions, new_im), (test_object.layer_indices, test_object.feature_indices))
      y1 = test_object.test_dnn_pred(np.array([new_im]))
      y2 = test_object.ensemble_dnn_pred(np.array([im]))
      if y1 != y2:
        found_count += 1

        adversarials.append([im, new_im])
        inp_ub=test_object.inp_ub
        save_adversarial_examples([new_im/(inp_ub*1.0), '{0}-adv-{1}'.format(len(adversarials), y1)], [im/(inp_ub*1.0), '{0}-original-{1}'.format(len(adversarials), y2)], None, nc_results.split('/')[0])
        d_adv=(np.count_nonzero(im-new_im))
        d_advs.append(d_adv)
        if len(d_advs)%100==0:
          print_adversarial_distribution(d_advs, nc_results.replace('.txt', '')+'-adversarial-distribution.txt', True)

    covered, not_covered=nc_report(cover_layers, test_object.layer_indices, test_object.feature_indices)
    
    f = open(nc_results, "a")
    f.write('NC-cover: {0} #test cases: {1} #adversarial examples: {2} #diff: {3} #layer: {4} #pos: {5}\n'.format(1.0 * covered / (covered + not_covered), len(test_cases), len(adversarials), d_adv, nc_layer.layer_index, nc_pos))
    f.close()

    if not_covered==0: break

