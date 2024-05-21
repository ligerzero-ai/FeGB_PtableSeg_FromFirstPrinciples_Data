import sys

from custodian.custodian import Custodian
from custodian.vasp.handlers import VaspErrorHandler, UnconvergedErrorHandler, NonConvergingErrorHandler, PositiveEnergyErrorHandler
from custodian.vasp.jobs import VaspJob

output_filename = "vasp.log"
handlers = [VaspErrorHandler(output_filename=output_filename), UnconvergedErrorHandler(), NonConvergingErrorHandler(), PositiveEnergyErrorHandler()]
jobs = [VaspJob(sys.argv[1:], output_file=output_filename, suffix = "",
                settings_override = [{"dict": "INCAR", "action": {"_set": {"NSW": 0, "LAECHG": True, "LCHARGE": True, "NELM": 300, "EDIFF": 1E-5}}}])]
c = Custodian(handlers, jobs, max_errors=10)
c.run()
