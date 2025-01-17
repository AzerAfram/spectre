// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <limits>
#include <memory>
#include <pup.h>
#include <string>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Framework/ActionTesting.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/Actions/Goto.hpp"
#include "Time/Actions/ChangeStepSize.hpp"
#include "Time/AdaptiveSteppingDiagnostics.hpp"
#include "Time/History.hpp"
#include "Time/Slab.hpp"
#include "Time/StepChoosers/Constant.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/Tags/AdaptiveSteppingDiagnostics.hpp"
#include "Time/Tags/HistoryEvolvedVariables.hpp"
#include "Time/Tags/IsUsingTimeSteppingErrorControl.hpp"
#include "Time/Tags/StepChoosers.hpp"
#include "Time/Tags/TimeStep.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Time/Tags/TimeStepper.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeStepRequest.hpp"
#include "Time/TimeSteppers/AdamsBashforth.hpp"
#include "Time/TimeSteppers/LtsTimeStepper.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeVector.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"

namespace {
// a silly step chooser that just always rejects, to test the step rejection
// control-flow.
struct StepRejector : public StepChooser<StepChooserUse::LtsStep> {
  using argument_tags = tmpl::list<>;
  using compute_tags = tmpl::list<>;
  using PUP::able::register_constructor;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
  WRAPPED_PUPable_decl_template(StepRejector);  // NOLINT
#pragma GCC diagnostic pop
  explicit StepRejector(CkMigrateMessage* /*unused*/) {}
  StepRejector() = default;
  explicit StepRejector(const double decrease) : decrease_(decrease) {}

  std::pair<TimeStepRequest, bool> operator()(const double last_step) const {
    return {{.size_goal = last_step * decrease_}, false};
  }

  bool uses_local_data() const override { return false; }
  bool can_be_delayed() const override { return true; }

  void pup(PUP::er& p) override { p | decrease_; }

 private:
  double decrease_ = 1.0;
};

PUP::able::PUP_ID StepRejector::my_PUP_ID = 0;

struct Var : db::SimpleTag {
  using type = double;
};

struct System {
  using variables_tag = Var;
};

struct NoOpLabel {};

template <typename Metavariables>
struct Component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using const_global_cache_tags =
      tmpl::list<Tags::ConcreteTimeStepper<LtsTimeStepper>>;
  using simple_tags =
      tmpl::list<Tags::TimeStepId, Tags::Next<Tags::TimeStepId>, Tags::TimeStep,
                 Tags::Next<Tags::TimeStep>, ::Tags::StepChoosers,
                 Tags::IsUsingTimeSteppingErrorControl,
                 Tags::AdaptiveSteppingDiagnostics,
                 Tags::HistoryEvolvedVariables<Var>,
                 typename System::variables_tag>;
  using compute_tags = time_stepper_ref_tags<LtsTimeStepper>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization,
                             tmpl::list<ActionTesting::InitializeDataBox<
                                 simple_tags, compute_tags>>>,
      Parallel::PhaseActions<
          Parallel::Phase::Testing,
          tmpl::list<Actions::ChangeStepSize<
                         typename Metavariables::step_choosers_to_use>,
                     ::Actions::Label<NoOpLabel>,
                     /*UpdateU action is required to satisfy internal checks of
                       `ChangeStepSize`. It is not used in the test.*/
                     Actions::UpdateU<System>>>>;
};

template <typename StepChoosersToUse = AllStepChoosers>
struct Metavariables {
  using step_choosers_to_use = StepChoosersToUse;
  using system = System;
  static constexpr bool local_time_stepping = true;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<StepChooser<StepChooserUse::LtsStep>,
                   tmpl::list<StepChoosers::Constant<StepChooserUse::LtsStep>,
                              StepRejector>>>;
  };
  using component_list = tmpl::list<Component<Metavariables>>;
};

template <typename StepChoosersToUse = AllStepChoosers>
void check(const bool time_runs_forward,
           std::unique_ptr<LtsTimeStepper> time_stepper,
           TimeSteppers::History<double> history, const Time& time,
           const double request, const TimeDelta& expected_step,
           std::optional<std::unique_ptr<StepRejector>> rejector) {
  CAPTURE(time);
  CAPTURE(request);

  const TimeDelta initial_step_size = (time_runs_forward ? 1 : -1) *
                                      time.slab().duration() /
                                      time.fraction().denominator();

  using component = Component<Metavariables<StepChoosersToUse>>;
  using MockRuntimeSystem =
      ActionTesting::MockRuntimeSystem<Metavariables<StepChoosersToUse>>;
  using Constant = StepChoosers::Constant<StepChooserUse::LtsStep>;
  MockRuntimeSystem runner{{std::move(time_stepper), 1e-8}};

  auto choosers =
      make_vector<std::unique_ptr<StepChooser<StepChooserUse::LtsStep>>>(
          std::make_unique<Constant>(2. * request),
          std::make_unique<Constant>(request),
          std::make_unique<Constant>(2. * request));
  if (rejector.has_value()) {
    choosers.emplace_back(std::move(*rejector));
  }

  // Initialize the component
  ActionTesting::emplace_component_and_initialize<component>(
      &runner, 0,
      {TimeStepId(time_runs_forward, 0,
                  time_runs_forward ? time.slab().start() : time.slab().end()),
       TimeStepId(time_runs_forward, 0, time), initial_step_size,
       initial_step_size, std::move(choosers), false,
       AdaptiveSteppingDiagnostics{1, 2, 3, 4, 5}, std::move(history), 1.});

  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);
  runner.template next_action<component>(0);
  const auto& box = ActionTesting::get_databox<component>(runner, 0);

  const size_t index =
      ActionTesting::get_next_action_index<component>(runner, 0);
  if (rejector.has_value()) {
    // if the step is rejected, it should jump to the UpdateU action
    CHECK(index == 2_st);
    CHECK(db::get<Tags::AdaptiveSteppingDiagnostics>(box) ==
          AdaptiveSteppingDiagnostics{1, 2, 3, 4, 6});
  } else {
    CHECK(index == 1_st);
    CHECK(db::get<Tags::AdaptiveSteppingDiagnostics>(box) ==
          AdaptiveSteppingDiagnostics{1, 2, 3, 4, 5});
  }
  CHECK(db::get<Tags::Next<Tags::TimeStep>>(box) == expected_step);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Time.Actions.ChangeStepSize", "[Unit][Time][Actions]") {
  register_classes_with_charm<TimeSteppers::AdamsBashforth>();
  register_factory_classes_with_charm<Metavariables<>>();
  const Slab slab(-5., -2.);
  const double slab_length = slab.duration().value();
  for (auto reject_step : {true, false}) {
    check(true, std::make_unique<TimeSteppers::AdamsBashforth>(1), {},
          slab.start() + slab.duration() / 4, slab_length / 5.,
          slab.duration() / 8,
          reject_step ? std::optional{std::make_unique<StepRejector>(0.5)}
                      : std::nullopt);
    check(true, std::make_unique<TimeSteppers::AdamsBashforth>(1), {},
          slab.start() + slab.duration() / 4, slab_length,
          reject_step ? slab.duration() / 8 : slab.duration() / 4,
          reject_step ? std::optional{std::make_unique<StepRejector>(0.5)}
                      : std::nullopt);
    check(false, std::make_unique<TimeSteppers::AdamsBashforth>(1), {},
          slab.end() - slab.duration() / 4, slab_length / 5.,
          -slab.duration() / 8,
          reject_step ? std::optional{std::make_unique<StepRejector>(0.5)}
                      : std::nullopt);
    check(false, std::make_unique<TimeSteppers::AdamsBashforth>(1), {},
          slab.end() - slab.duration() / 4, slab_length,
          reject_step ? -slab.duration() / 8 : -slab.duration() / 4,
          reject_step ? std::optional{std::make_unique<StepRejector>(0.5)}
                      : std::nullopt);

    // Check for roundoff issues
    check(true, std::make_unique<TimeSteppers::AdamsBashforth>(1), {},
          slab.start() + slab.duration() / 4,
          slab_length / 16. / (1.0 + std::numeric_limits<double>::epsilon()),
          slab.duration() / 32,
          reject_step ? std::optional{std::make_unique<StepRejector>(0.5)}
                      : std::nullopt);
    check(false, std::make_unique<TimeSteppers::AdamsBashforth>(1), {},
          slab.end() - slab.duration() / 4,
          slab_length / 16. / (1.0 + std::numeric_limits<double>::epsilon()),
          -slab.duration() / 32,
          reject_step ? std::optional{std::make_unique<StepRejector>(0.5)}
                      : std::nullopt);
  }

  {
    // History out of order, as if just after self-start.
    TimeSteppers::History<double> history{};
    history.insert(TimeStepId(true, -1, slab.start() + slab.duration() / 8),
                   0.0, 0.0);
    check(true, std::make_unique<TimeSteppers::AdamsBashforth>(1),
          std::move(history), slab.start() + slab.duration() / 4, 1.0e-3,
          slab.duration() / 4, std::nullopt);
  }

  CHECK_THROWS_WITH(
      ([&slab, &slab_length]() {
        check<tmpl::list<StepChoosers::Constant<StepChooserUse::LtsStep>>>(
            true, std::make_unique<TimeSteppers::AdamsBashforth>(1), {},
            slab.start() + slab.duration() / 4, slab_length / 5.,
            slab.duration() / 8, std::make_unique<StepRejector>(0.5));
      })(),
      Catch::Matchers::ContainsSubstring("is not registered"));

#ifdef SPECTRE_DEBUG
  CHECK_THROWS_WITH(
      check(true, std::make_unique<TimeSteppers::AdamsBashforth>(1), {},
            slab.start() + slab.duration() / 4, slab_length,
            slab.duration() / 4, std::make_unique<StepRejector>(1.0)),
      Catch::Matchers::ContainsSubstring("Step was rejected, but not changed"));
#endif

  CHECK_THROWS_WITH(
      check(true, std::make_unique<TimeSteppers::AdamsBashforth>(1), {},
            slab.start() + slab.duration() / 4, 1e-9, slab.duration() / 4,
            std::nullopt),
      Catch::Matchers::ContainsSubstring("smaller than the MinimumTimeStep"));
}
