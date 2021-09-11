

import gzip
import re, glob
import logging
from collections import defaultdict
from SmallFuction
import numpy as np
import threading
import colors
import weakref
import _pickle as pickle
from mkdir import mkdir
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
# from matplotlib import colors
from shapely.geometry import asPolygon

# Display more info for debug
debug_mode = True

dic_orientation = {
    'E': lambda loc: (-loc[1], loc[0]),
    'N': lambda loc: (loc[0], loc[1]),
    'FN': lambda loc: (-loc[0], loc[1]),
    'S': lambda loc: (-loc[0], -loc[1]),
    'FS': lambda loc: (loc[0], -loc[1]),
    'FE': lambda loc: (loc[1], loc[0]),
    'W': lambda loc: (-loc[1], loc[0]),
    'FW': lambda loc: (-loc[1], -loc[0]),
}

dic_orientation_reversed = {
    'E': lambda loc: (-loc[1], loc[0]),
    'N': lambda loc: (loc[0], loc[1]),
    'FN': lambda loc: (-loc[0], loc[1]),
    'S': lambda loc: (-loc[0], -loc[1]),
    'FS': lambda loc: (loc[0], -loc[1]),
    'FE': lambda loc: (loc[1], loc[0]),
    'W': lambda loc: (loc[1], -loc[0]),
    'FW': lambda loc: (-loc[1], -loc[0]),
}


class TileMaster(object):
    """
    **Tile Ref Name of Master**
    * ref_name   : The ref name of tile
    * color      :
    * edges      : Store edges
    * allow edge : Store edges allowing feed through
    * origin     : Store left bottom and top right origin
    * pitch      : Metal pitch
    """
    def __init__(self, ref_name, color, ref_poly):
        self.ref_name   = ref_name
        self.color      = color
        self.ref_poly   = ref_poly
        self.edges      = {}
        self.allow_edge = {}
        self.origin     = ()
        self.pitch      = {'M4': {'X': [('start', 'do', 'pitch')], 'Y': [('start', 'do', 'pitch')]}}

class TileInstance(object):
    """
    **Tile Instantiation  **
    * inst_name   :
    * abut_inst   :
    * edges       :
    * allow_edge  :
    * reuse_list  :
    * inst_ploy   :
    * inst_loc    :
    * inst_size   :     (width, height)
    * inst_orient :
    * tile_master :
    """
    def __init__(self, tile_master_class):
        self.inst_name      = ''
        self.abut_inst      = []
        self.edges          = {}
        self.allow_edge     = {}
        self.reuse_list     = []
        self.inst_ploy      = []
        self.inst_loc       = []
        self.inst_size      = []
        self.inst_orient    = ''
        self.tile_master    = tile_master_class

    def __str__(self):
        return '(Tile:%s %s,)' % (self.ref_name, self.inst_name)

class chip(object):
    def __init__(self):
        self.dic_tiles = {}


class Top(chip):
    def __init__(self):
        """
        * initial params of Top
        """
        super(Top, self).__init__()
        self.dic_inst2master    = {}
        self.dic_master2inst    = defaultdict(list)
        self.dic_tile2poly      = defaultdict(list)
        self.dic_tile2rect      = {}
        self.dic_params         = {}
        self.dic_inst2master    = {}
        self.dic_ref2inst       = {}
        self.dic_inst2ref       = {}
        self.dic_tile2track     = {}
        self.LOG = open('Top.log', 'w+')

    def getReuse(self):
        """
        format of self.dic_inst2master:
        #Tile:[Tile_reuse0,...,Tile_reuseN]
        format of self.dic_master2inst:
        #Tile_reuse0:Tile,....Tile_reuseN:Tile
        """
        self.dic_inst2master    = {}
        self.dic_master2inst    = defaultdict(list)

        for tile in self.dic_tiles:
            self.dic_inst2master[self.dic_tiles[tile].inst_name] = self.dic_tiles[tile].tile_master.ref_name
            self.dic_master2inst[self.dic_tiles[tile].tile_master.ref_name].append(self.dic_tiles[tile].inst_name)

    def parseCompdef (self, file="", read_from_pkl=False):
        """
        Parse def‘s components
        file = "data/Floorplan/*/*def.gz"
        """
        dic_inst2master = {}
        def_file = glob.glob(file)
        regx_inst = re.compile(r'- (\S+) (\S+).*\( (.*) \) (\S+) ;')

        #if read_from_pkl:
        #    with mkdir(file_name='getshape.pkl', mode='rb', type='data') as file:
        #        def_file = []
        #        db = pickle.load(file)
        #        self.dic_inst2master = db['dic_inst2master']
        #        self.dic_tile2poly = db['dic_tile2poly']
        #    # self.dic_tile2track= db['dic_tile2track']

        for file in def_file:
            with gzip.open(file) as f:
                for line in f:
                    line_str = line.decode()
                    if 'END COMPONENTS' in line_str: break
                    mt_inst = regx_inst.search(line_str)
                    if mt_inst:
                        tmp = mt_inst.groups()
                        dic_inst2master[tmp[0]] = [tmp[1], (
                        float(tmp[2].split()[0]) / 2000, float(tmp[2].split()[1]) / 2000), tmp[-1]]
                        #print (tmp[0],dic_inst2master[tmp[0]])
        return dic_inst2master

    def parserbboxdef(self,file=""):
        dic_tile2poly = defaultdict(list)
        regx_diearea = re.compile(r'DIEAREA (\(.*\)) ;')
        regx_design = re.compile(r'DESIGN (\S+) ;')
        regx_tuple = re.compile(r'\( (\S+ \S+) \)')
        regx_track = re.compile(r'TRACKS (Y|X) (\S+) DO (\S+) STEP (\S+) LAYER (M4|M5|M6|M7|M8)')
        def_file = glob.glob(file)
        for file in def_file:
            with gzip.open(file) as f:
                track_info = defaultdict(list)
                for line in f:
                   line_str = line.decode()
                   mt_design = regx_design.search(line_str)
                   mt_diearea = regx_diearea.search(line_str)
                   mt_track = regx_track.search(line_str)
                   if mt_design:
                       tile = mt_design.groups()[0]
                       if tile in dic_tile2poly: break
                   if mt_diearea:
                       area = mt_diearea.groups()[0]
                       print (tile,area,file=self.LOG)
                       mt_tuple = regx_tuple.findall(area)
                       if len(mt_tuple) != 2:
                           maxX, maxY = (0, 0)
                           minX, minY = (0, 0)
                           for i in mt_tuple:
                               x, y = [int(j) for j in i.split()]
                               if x > maxX: maxX = x
                               if y > maxY: maxY = y
                               if x < minX: minX = x
                               if y < minY: minY = y
                               x_center, y_center = float(maxX + minX) / 4000, float(maxY + minY) / 4000
                               dic_tile2poly[tile].append((float(x) / 2000, float(y) / 2000))
                           for id in range(len(dic_tile2poly[tile])):
                               i = dic_tile2poly[tile][id]
                               dic_tile2poly[tile][id] = (i[0] - x_center, i[1] - y_center)
                       elif len(mt_tuple) == 2:
                           # change bbox to clockwise polygon if diearea is bbox type
                           x0, y0 = [int(j) for j in mt_tuple[0].split()]
                           x1, y1 = [int(j) for j in mt_tuple[1].split()]
                           x_center, y_center = float(x0 + x1) / 4000, float(y0 + y1) / 4000
                           dic_tile2poly[tile].append(
                               (float(x0) / 2000 - x_center, float(y0) / 2000 - y_center))
                           dic_tile2poly[tile].append(
                               (float(x0) / 2000 - x_center, float(y1) / 2000 - y_center))
                           dic_tile2poly[tile].append(
                               (float(x1) / 2000 - x_center, float(y1) / 2000 - y_center))
                           dic_tile2poly[tile].append(
                               (float(x1) / 2000 - x_center, float(y0) / 2000 - y_center))
        return dic_tile2poly

    def tile2inst(self):
        db = {}
        db['dic_tile2poly'] = self.dic_tile2poly
        db['dic_inst2master'] = self.dic_inst2master
        #       print (self.dic_tile2poly)
        with mkdir(file_name='getshape.pkl', mode='wb', type='data') as file:
            pickle.dump(db, file)
        # Generate color for master tiles
        c = colors.colors()
        dic_master2color = {master: c.get_random_color() for master in self.dic_tile2poly}

        # Generate meta class for tile master
        dic_meta2tile = {}
        for master in dic_master2color:
            ref_poly = np.array(self.dic_tile2poly.get(master, [[-5, -5], [5, -5], [5, 5], [-5, 5]]))
            shape = ref_poly.shape[0]
            edges = {}
            for p in range(shape):
                edge_key = (ref_poly[p % shape][0], ref_poly[p % shape][1], ref_poly[(p + 1) % shape][0],
                            ref_poly[(p + 1) % shape][1])
                edges[edge_key] = edge_key

            x_min, y_min = ref_poly.min(axis=0)
            x_max, y_max = ref_poly.max(axis=0)
            origin = (x_min, y_min, x_max, y_max)
            dic_meta2tile[master] = type(master, (), {'edges': edges,
                                                      'allowedge': edges,
                                                      'ref_name': master,
                                                      'color': dic_master2color[master],
                                                      'ref_poly': ref_poly,
                                                      'origin': origin,
                                                      # 'track':self.dic_tile2track[master]
                                                      })

        # Generate class for chip
        # CHIP->Master->Instance
        for inst in self.dic_inst2master:
            master = self.dic_inst2master[inst][0]
            if master not in self.dic_tile2poly: continue
            self.dic_tiles[inst] = Tile_Instance(dic_meta2tile[master])
            self.dic_tiles[inst].inst_name = inst
            self.dic_tiles[inst].inst_loc = self.dic_inst2master[inst][1]
            self.dic_tiles[inst].inst_orient = self.dic_inst2master[inst][2]
            x_min, y_min, x_max, y_max = self.dic_tiles[inst].tile_master.origin
            self.dic_tiles[inst].inst_size = (x_max - x_min, y_max - y_min)
            tmp1 = np.array(
                [[abs(x_min) + self.dic_tiles[inst].inst_loc[0], abs(y_min) + self.dic_tiles[inst].inst_loc[1]] for i in
                 range(self.dic_tiles[inst].tile_master.ref_poly.shape[0])])
            tmp2 = np.array([dic_orientation[self.dic_tiles[inst].inst_orient](point) for point in
                             self.dic_tiles[inst].tile_master.ref_poly])
            tmp = tmp1 + tmp2
            self.dic_tiles[inst].inst_poly = tmp

            # generate edge relationship between instance and master tile
            tmp3 = np.array([point for point in self.dic_tiles[inst].tile_master.ref_poly])
            shape = tmp2.shape[0]
            self.dic_inst2ref[inst] = {
                (tmp[p % shape][0], tmp[p % shape][1], tmp[(p + 1) % shape][0], tmp[(p + 1) % shape][1]): (
                tmp3[p % shape][0], tmp3[p % shape][1], tmp3[(p + 1) % shape][0], tmp3[(p + 1) % shape][1]) for p in
                range(shape)}
            # reverse key & value
            self.dic_ref2inst[inst] = {v: k for k, v in self.dic_inst2ref[inst].items()}

    def parsetopdef(self,file="", read_from_pkl=False):
        def_files = glob.glob(file)
        for file in def_files:
            if 'chip.def' in file:
                chip_inst2master = self.parsecompdef(file,read_from_pkl=False)
            elif 'ce_c.def' in file:
                ce_inst2master = self.parsecompdef(file)
                self.dic_tile2poly.update(self.parserbboxdef(file))
            else:
                self.dic_tile2poly.update(self.parserbboxdef(file))
        for inst in chip_inst2master:
            if 'ce_c' in inst:
                print (self.dic_tile2poly['ce_c'])
                #loc = chip_inst2master[inst][1]
                ce_loc = chip_inst2master[inst][1]
                for ce_inst in ce_inst2master:
                    ce_inst_loc = ce_inst2master[ce_inst][1]
                    loc = (float(ce_loc[0])+float(ce_inst_loc[0]+abs(self.dic_tile2poly['ce_c'][0][0])),
                           float(ce_loc[1])+float(ce_inst_loc[1])+abs(self.dic_tile2poly['ce_c'][0][1]))
                    ce_inst_name = inst+"_"+ce_inst
                    #print ([ce_inst_name,ce_inst2master[ce_inst][0],loc,chip_inst2master[inst][2]])
                    self.dic_inst2master[ce_inst_name] = [ce_inst2master[ce_inst][0],loc,chip_inst2master[inst][2]]
            self.dic_inst2master[inst] = chip_inst2master[inst]

        print (self.dic_inst2master)
        self.tile2inst()

    def plot2(self, special=[]):
        fig, ax = plt.subplots()
        for inst in self.dic_tiles:
            if self.dic_tiles[inst].inst_poly.shape[0] == 1: continue
            tmp = np.row_stack((self.dic_tiles[inst].inst_poly, np.array(self.dic_tiles[inst].inst_poly[0])))
            vertices = np.array(tmp, float)
            codes = [Path.MOVETO] + [Path.LINETO] * (self.dic_tiles[inst].inst_poly.shape[0] - 1) + [Path.CLOSEPOLY]
            path = Path(vertices, codes)
            edgecolor = 'red' if inst in special else 'black'
            if len(special):
                if inst in special:
                    facecolor = 'blue'
                else:
                    facecolor = 'lightslategray'
            else:
                facecolor = self.dic_tiles[inst].tile_master.color
                alpha_v = 0.8
            if 'ce_c' in inst:
                alpha_v = 0.1
            pathpatch = PathPatch(path, facecolor=facecolor, edgecolor=edgecolor,alpha=alpha_v)

            ax.add_patch(pathpatch)
            # ax.dataLim.update_from_data_xy(vertices)
        ax.autoscale_view()
        plt.show()

    def __del__(self):
        self.LOG.close()

if __name__ == "__main__":
    import datetime

    begin = datetime.datetime.now()
    tmp = Top()
    tmp.parsetopdef(file="xinzhou/tile_defs/*.def.gz", read_from_pkl=False)
    tmp.getreuse()
    end = datetime.datetime.now()
    tmp.plot2()
    print(end - begin)
