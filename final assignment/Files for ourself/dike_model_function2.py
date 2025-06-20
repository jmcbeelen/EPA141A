"""
Created on Tue Oct 31 13:18:05 2017

@author: ciullo
"""
import copy
import numpy as np
import pandas as pd
from collections import defaultdict

from ema_workbench import ema_logging

import funs_generate_network
from funs_dikes import Lookuplin, dikefailure, init_node
from funs_economy import cost_fun, discount, cost_evacuation
from funs_hydrostat import werklijn_cdf, werklijn_inv


def Muskingum(C1, C2, C3, Qn0_t1, Qn0_t0, Qn1_t0):
    """Simulates hydrological routing"""
    Qn1_t1 = C1 * Qn0_t1 + C2 * Qn0_t0 + C3 * Qn1_t0
    return Qn1_t1


class DikeNetwork:
    def __init__(self):
        # planning steps
        self.num_planning_steps = 3
        self.num_events = 30

        # load network
        G, dike_list, dike_branch, planning_steps = funs_generate_network.get_network(
            self.num_planning_steps
        )

        # Load hydrological statistics:
        self.A = pd.read_excel("./data/hydrology/werklijn_params.xlsx")

        lowQ, highQ = werklijn_inv([0.992, 0.99992], self.A)
        self.Qpeaks = np.unique(
            np.asarray(
                [np.random.uniform(lowQ, highQ) / 6 for _ in range(0, self.num_events)]
            )
        )[::-1]

        # Probabiltiy of exceedence for the discharge @ Lobith (i.e. times 6)
        self.p_exc = 1 - werklijn_cdf(self.Qpeaks * 6, self.A)

        self.G = G
        self.dikelist = dike_list
        self.dike_branch = dike_branch
        self.planning_steps = planning_steps
        self.hri_ts = []

        # Accounting for the discharge reduction due to upstream dike breaches
        self.sb = True

        # Planning window [y], reasonable for it to be a multiple of num_planning_steps
        self.n = 200
        # Years in planning step:
        self.y_step = self.n // self.num_planning_steps
        # Step of dike increase [m]
        self.dh = 0.1

        # Time step correction: Q is a mean daily value expressed in m3/s
        self.timestepcorr = 24 * 60 * 60

    #        ema_logging.info('model initialized')

    # Initialize hydrology at each node:
    def _initialize_hydroloads(self, node, time, Q_0):
        node["cumVol"], node["wl"], node["Qpol"], node["hbas"] = (
            init_node(0, time) for _ in range(4)
        )
        node["Qin"], node["Qout"] = (init_node(Q_0, time) for _ in range(2))
        node["status"] = init_node(False, time)
        node["tbreach"] = np.nan
        return node

    def _initialize_rfr_ooi(self, G, dikenodes, steps):
        for s in steps:
            for n in dikenodes:
                node = G.nodes[n]
                # Create a copy of the rating curve that will be used in the sim:
                node["rnew"] = copy.deepcopy(node["r"])

                # Initialize outcomes of interest (ooi):
                node[f"losses {s}"] = []
                node[f"deaths {s}"] = []
                node[f"evacuation_costs {s}"] = []

            # Initialize room for the river
            G.nodes[f"RfR_projects {s}"]["cost"] = 0
        return G

    def progressive_height_and_costs(self, G, dikenodes, steps):
        for dike in dikenodes:
            node = G.nodes[dike]
            # Rescale according to step and tranform in meters
            for s in steps:
                node[f"DikeIncrease {s}"] *= self.dh
                # 1 Initialize fragility curve
                # 2 Shift it to the degree of dike heigthening:
                # 3 Calculate cumulative raising

                node[f"fnew {s}"] = copy.deepcopy(node["f"])
                node[f"dikeh_cum {s}"] = 0

                for ss in steps[steps <= s]:
                    node[f"fnew {s}"][:, 0] += node[f"DikeIncrease {ss}"]
                    node[f"dikeh_cum {s}"] += node[f"DikeIncrease {ss}"]

                # Calculate dike heigheting costs:
                if node[f"DikeIncrease {s}"] == 0:
                    node[f"dikecosts {s}"] = 0
                else:
                    node[f"dikecosts {s}"] = cost_fun(
                        node["traj_ratio"],
                        node["c"],
                        node["b"],
                        node["lambda"],
                        node[f"dikeh_cum {s}"],
                        node[f"DikeIncrease {s}"],
                    )

    def __call__(self, timestep=1, **kwargs):

        G = copy.deepcopy(self.G)
        Qpeaks = self.Qpeaks
        dikelist = self.dikelist
        self.hri_ts.clear() #added to ensure fresh list per run

        # Call RfR initialization:
        self._initialize_rfr_ooi(G, dikelist, self.planning_steps)

        # Load all kwargs into network. Kwargs are uncertainties and levers:
        for item in kwargs:
            # when item is 'discount rate':
            if "discount rate" in item:
                G.nodes[item]["value"] = kwargs[item]
            # the rest of the times you always get a string like {}_{}:
            else:
                string1, string2 = item.split("_")

                if "RfR" in string2:
                    # string1: projectID
                    # string2: rfr #step
                    # Note: kwargs[item] in this case can be either 0
                    # (no project) or 1 (yes project)
                    temporal_step = string2.split(" ")[1]

                    proj_node = G.nodes[f"RfR_projects {temporal_step}"]
                    # Cost of RfR project
                    proj_node["cost"] += (
                        kwargs[item] * proj_node[string1]["costs_1e6"] * 1e6
                    )

                    # Iterate over the location affected by the project
                    for key in proj_node[string1].keys():
                        if key != "costs_1e6":
                            # Change in rating curve due to the RfR project
                            G.nodes[key]["rnew"][:, 1] -= (
                                kwargs[item] * proj_node[string1][key]
                            )
                else:
                    # string1: dikename or EWS
                    # string2: name of uncertainty or lever
                    G.nodes[string1][string2] = kwargs[item]

        self.progressive_height_and_costs(G, dikelist, self.planning_steps)

        # Percentage of people who can be evacuated for a given warning
        # time:
        G.nodes["EWS"]["evacuation_percentage"] = G.nodes["EWS"]["evacuees"][
            G.nodes["EWS"]["DaysToThreat"]
        ]

        # Dictionary storing outputs:
        data = defaultdict(list)

        #added
        for dike in self.dikelist:
            data[f"{dike}_Hydrological Resilience Index"] = []  # per-dike series
        # system-wide series container
        data["Hydrological Resilience Index"] = []

        for s in self.planning_steps:
            for Qpeak in Qpeaks:
                node = G.nodes["A.0"]
                waveshape_id = node["ID flood wave shape"]

                time = np.arange(
                    0, node["Qevents_shape"].loc[waveshape_id].shape[0], timestep
                )
                node["Qout"] = Qpeak * node["Qevents_shape"].loc[waveshape_id]

                # Initialize hydrological event:
                for key in dikelist:
                    node = G.nodes[key]

                    Q_0 = int(G.nodes["A.0"]["Qout"][0])

                    self._initialize_hydroloads(node, time, Q_0)
                    # Calculate critical water level: water above which failure
                    # occurs
                    node["critWL"] = Lookuplin(node[f"fnew {s}"], 1, 0, node["pfail"])

                # Run the simulation:
                # Run over the discharge wave:
                for t in range(1, len(time)):
                    # Run over each node of the branch:
                    for n in range(0, len(dikelist)):
                        # Select current node:
                        node = G.nodes[dikelist[n]]
                        if node["type"] == "dike":

                            # Muskingum parameters:
                            C1 = node["C1"]
                            C2 = node["C2"]
                            C3 = node["C3"]

                            prec_node = G.nodes[node["prec_node"]]
                            # Evaluate Q coming in a given node at time t:
                            node["Qin"][t] = Muskingum(
                                C1,
                                C2,
                                C3,
                                prec_node["Qout"][t],
                                prec_node["Qout"][t - 1],
                                node["Qin"][t - 1],
                            )

                            # Transform Q in water levels:
                            node["wl"][t] = Lookuplin(
                                node["rnew"], 0, 1, node["Qin"][t]
                            )

                            # Evaluate failure and, in case, Q in the floodplain and
                            # Q left in the river:
                            res = dikefailure(
                                self.sb,
                                node["Qin"][t],
                                node["wl"][t],
                                node["hbas"][t],
                                node["hground"],
                                node["status"][t - 1],
                                node["Bmax"],
                                node["Brate"],
                                time[t],
                                node["tbreach"],
                                node["critWL"],
                            )

                            node["Qout"][t] = res[0]
                            node["Qpol"][t] = res[1]
                            node["status"][t] = res[2]
                            node["tbreach"] = res[3]

                            # Evaluate the volume inside the floodplain as the integral
                            # of Q in time up to time t.
                            node["cumVol"][t] = (
                                np.trapz(node["Qpol"]) * self.timestepcorr
                            )

                            Area = Lookuplin(node["table"], 4, 0, node["wl"][t])
                            node["hbas"][t] = node["cumVol"][t] / float(Area)

                        elif node["type"] == "downstream":
                            node["Qin"] = G.nodes[dikelist[n - 1]]["Qout"]

                # Iterate over the network and store outcomes of interest for a
                # given event
                for dike in self.dikelist:
                    # data[f"{dike}_Hydrological Resilience Index"] = []
                    node = G.nodes[dike]

                    # If breaches occured:
                    if node["status"][-1] == True:
                        # Losses per event:
                        node[f"losses {s}"].append(
                            Lookuplin(node["table"], 6, 4, np.max(node["wl"]))
                        )

                        node[f"deaths {s}"].append(
                            Lookuplin(node["table"], 6, 3, np.max(node["wl"]))
                            * (1 - G.nodes["EWS"]["evacuation_percentage"])
                        )

                        node[f"evacuation_costs {s}"].append(
                            cost_evacuation(
                                Lookuplin(node["table"], 6, 5, np.max(node["wl"]))
                                * G.nodes["EWS"]["evacuation_percentage"],
                                G.nodes["EWS"]["DaysToThreat"],
                            )
                        )
                    else:
                        node[f"losses {s}"].append(0)
                        node[f"deaths {s}"].append(0)
                        node[f"evacuation_costs {s}"].append(0)

            # data["Hydrological Resilience Index"] = []
            EECosts = []
            # Iterate over the network,compute and store ooi over all events
            for dike in dikelist:
                node = G.nodes[dike]

                # Expected Annual Damage:
                EAD = np.trapz(node[f"losses {s}"], self.p_exc)
                # Discounted annual risk per dike ring:
                disc_EAD = np.sum(
                    discount(
                        EAD, rate=G.nodes[f"discount rate {s}"]["value"], n=self.y_step
                    )
                )

                # Expected Annual number of deaths:
                END = np.trapz(node[f"deaths {s}"], self.p_exc)

                # Expected Evacuation costs: depend on the event, the higher
                # the event, the more people you have got to evacuate:
                EECosts.append(np.trapz(node[f"evacuation_costs {s}"], self.p_exc))

                data[f"{dike}_Expected Annual Damage"].append(disc_EAD)
                data[f"{dike}_Expected Number of Deaths"].append(END)
                data[f"{dike}_Dike Investment Costs"].append(node[f"dikecosts {s}"])

            data[f"RfR Total Costs"].append(G.nodes[f"RfR_projects {s}"]["cost"])
            data[f"Expected Evacuation Costs"].append(np.sum(EECosts))

        #hier begint h

        # Bereken HRI en GRSI per dijkring, en systeem-breed

        rfr_year_weights = {0: 1.0, 1: 0.75, 2: 0.5, 3: 0.25}
        max_rfr = 5
        max_fat = 5
        max_dike = 10
        w1, w2, w3, w4 = 1, 1, 1, 1

        # Berekeningen per dijk
        for dike in self.dikelist:
            total_dike_increase = 0
            rfr_coverage = 0
            fat = 0
            ead = 0

            for s in self.planning_steps:
                node = G.nodes[dike]
                total_dike_increase += node[f"DikeIncrease {s}"]

                for rfr_node in G.nodes:
                    if rfr_node.startswith("RfR_projects"):
                        step_num = int(rfr_node.split(" ")[-1])
                        cost = G.nodes[rfr_node]["cost"]
                        if cost > 0:
                            rfr_coverage += 1
                            fat += rfr_year_weights.get(step_num, 0)

                # Pak alleen schade voor deze dijk
                if f"{dike}_Expected Annual Damage" in data:
                    ead = data[f"{dike}_Expected Annual Damage"][-1]
                    flood_penalty = 1 if ead > 1e7 else 0

                # Normaliseer en bereken HRI
                hri = (
                        w1 * (rfr_coverage / max_rfr)
                        + w2 * (fat / max_fat)
                        - w3 * (total_dike_increase / max_dike)
                        - w4 * flood_penalty
                            )
                data[f"{dike}_Hydrological Resilience Index"].append(hri)
                print("DEBUG HRI append", dike, s, hri)

                # system-wide average for this planning step
        step_hris = [
            data[f"{d}_Hydrological Resilience Index"][-1] for d in self.dikelist
        ]
        system_hri = np.mean(step_hris)
        data["Hydrological Resilience Index"].append(system_hri)









        # # === Systeem-brede HRI ===
        # all_hris = [data[f"{dike}_Hydrological Resilience Index"] for dike in self.dikelist if
        #             f"{dike}_Hydrological Resilience Index" in data]
        #data["Hydrological Resilience Index"] = np.mean(all_hris) if all_hris else 0

        # === Totale uitstroom per dijk ===
        for dike in self.dikelist:
            Qin = G.nodes[dike].get("Qin", [])
            Qout = G.nodes[dike].get("Qout", [])
            Qpol = G.nodes[dike].get("Qpol", [])

            total_Qin = np.sum(Qin)
            total_Qout = np.sum(Qout)
            total_Qpol = np.sum(Qpol)

            data[f"{dike}_Total Inflow"] = total_Qin
            data[f"{dike}_Total Outflow"] = total_Qout
            data[f"{dike}_Total Breach Flow"] = total_Qpol

        # === Systeem-breed totaal (optioneel) ===
        data["System Total Inflow"] = sum([data[f"{d}_Total Inflow"] for d in self.dikelist])
        data["System Total Outflow"] = sum([data[f"{d}_Total Outflow"] for d in self.dikelist])
        data["System Total Breach Flow"] = sum([data[f"{d}_Total Breach Flow"] for d in self.dikelist])

        return data


