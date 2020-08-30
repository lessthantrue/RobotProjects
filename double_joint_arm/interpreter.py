from pygcode import *
import os

class Interpreter():
    def __init__(self):
        self.file = None
        self.mac = Machine()
        self.planner = None
        self.fr = 0
        self.EOF = False

    def setPlanner(self, planner):
        self.planner = planner

    def loadFile(self, fname):
        if not (self.file is None):
            self.file.close()
        self.file = open(fname, "r")

    # returns true if a movement needs to be made
    def nextLine(self):
        line_txt = self.file.readline()
        line = Line(line_txt)
        if line_txt == "":
            # exit on end of file to return control to the outer simulation loop
            self.EOF = True
            return True
        block = self.mac.block_modal_gcodes(line.block)
        xb, yb, zb = self.mac.pos.vector
        self.mac.process_gcodes(*sorted(block))
        xa, ya, za = self.mac.pos.vector
        for b in block:
            if isinstance(b, GCodeLinearMove):
                self.planner.G1(self.mac.pos.vector[0], self.mac.pos.vector[1], self.fr)
                return True
            elif isinstance(b, GCodeRapidMove):
                self.planner.G0(self.mac.pos.vector[0], self.mac.pos.vector[1])
                return True
            elif isinstance(b, GCodeFeedRate):
                self.fr = b.word.value
            elif isinstance(b, GCodeArcMoveCW):
                # I don't know how any of this works, just copied from a pygcode arc linearizing example
                arc_center_ijk = dict((l, 0.) for l in "IJK")
                arc_center_ijk.update(b.get_param_dict("IJK"))
                arc_center_coords = dict(({"I":"x", "J":"y", "K":"z"}[k], v) for (k, v) in arc_center_ijk.items())
                # print(arc_center_coords)
                # incremental position
                # problem is here: xc and yc are only offset by position if machine is in incremental mode
                # also needs to be offset from tool at start of instruction, not where the machine is afterwards.
                xc = arc_center_coords['x'] + xb
                yc = arc_center_coords['y'] + yb
                self.planner.G2(self.mac.pos.vector[0], self.mac.pos.vector[1], xc ,yc, self.fr)
                return True
            elif isinstance(b, GCodeArcMoveCCW):
                arc_center_ijk = dict((l, 0.) for l in "IJK")
                arc_center_ijk.update(b.get_param_dict("IJK"))
                arc_center_coords = dict(({"I":"x", "J":"y", "K":"z"}[k], v) for (k, v) in arc_center_ijk.items())
                xc = arc_center_coords['x'] + xb
                yc = arc_center_coords['y'] + yb
                self.planner.G3(self.mac.pos.vector[0], self.mac.pos.vector[1], xc, yc, self.fr)
                return True
            else:
                print("Unknown Instruction: ", b)
        return False
        
