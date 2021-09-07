#
# Copyright (C) 2020 GreenWaves Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import gsystree as st

class Gvsoc(st.Component):

    def __init__(self, parent, name):

        super(Gvsoc, self).__init__(parent, name)

        self.add_property("proxy/enabled", False)
        self.add_property("proxy/port", 42951)

        self.add_property("events/enabled", False)
        self.add_property("events/include_raw", [])
        self.add_property("events/include_regex", [])
        self.add_property("events/exclude_regex", [])
        self.add_property("events/format", "fst")
        self.add_property("events/active", False)
        self.add_property("events/all", True)
        self.add_property("events/gtkw", False)
        self.add_property("events/gen_gtkw", True)
        self.add_property("events/files", [ ])
        self.add_property("events/traces", {})
        self.add_property("events/tags", [ "overview" ])
        self.add_property("events/gtkw", False)

        self.add_properties({
            "description": "GAP simulator.",

            "runner_module": "gv.gvsoc",
        
            "cycles_to_seconds": "int(max(cycles * nb_cores / 5000000, 600))",
        
            "verbose": True,
            "debug-mode": False,
            "sa-mode": True,
        
            "launchers": {
                "default": "gvsoc_launcher",
                "debug": "gvsoc_launcher_debug"
            },
        
            "traces": {
                "level": "debug",
                "format": "long",
                "enabled": False,
                "include_regex": [],
                "exclude_regex": []
            }
        })
