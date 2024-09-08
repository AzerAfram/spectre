// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Parallel/Algorithms/AlgorithmSingleton.hpp"
#include "Parallel/ArrayCollection/IsDgElementCollection.hpp"
#include "Parallel/ArrayCollection/SimpleActionOnElement.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/Phase.hpp"
#include "ParallelAlgorithms/Amr/Actions/AdjustDomain.hpp"
#include "ParallelAlgorithms/Amr/Actions/EvaluateRefinementCriteria.hpp"
#include "ParallelAlgorithms/Amr/Criteria/Tags/Criteria.hpp"
#include "ParallelAlgorithms/Amr/Policies/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \ingroup AmrGroup
namespace amr {
/// \brief A singleton parallel component to manage adaptive mesh refinement
///
/// \details This component can be used for:
/// - Running actions that create new elements.  This may be necessary to
///   work around Charm++ bugs, and may require the singleton to be placed
///   on global processor 0.
/// - As a reduction target to perform sanity checks after AMR, output
///   AMR diagnostics, or determine when to trigger AMR.
template <class Metavariables>
struct Component {
  using metavariables = Metavariables;

  using chare_type = Parallel::Algorithms::Singleton;

  using const_global_cache_tags =
      tmpl::list<amr::Criteria::Tags::Criteria, amr::Tags::Policies,
                 logging::Tags::Verbosity<amr::OptionTags::AmrGroup>>;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization, tmpl::list<>>>;

  using simple_tags_from_options = Parallel::get_simple_tags_from_options<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;

  static void execute_next_phase(
      const Parallel::Phase next_phase,
      Parallel::CProxy_GlobalCache<Metavariables>& global_cache_proxy) {
    auto& local_cache = *Parallel::local_branch(global_cache_proxy);
    Parallel::get_parallel_component<Component>(local_cache)
        .start_phase(next_phase);
    if (Parallel::Phase::EvaluateAmrCriteria == next_phase) {
      if constexpr (Parallel::is_dg_element_collection_v<
                        typename metavariables::amr::element_array>) {
        Parallel::threaded_action<Parallel::Actions::SimpleActionOnElement<
            ::amr::Actions::EvaluateRefinementCriteria, true>>(
            Parallel::get_parallel_component<
                typename metavariables::amr::element_array>(local_cache));
      } else {
        Parallel::simple_action<::amr::Actions::EvaluateRefinementCriteria>(
            Parallel::get_parallel_component<
                typename metavariables::amr::element_array>(local_cache));
      }
    }
    if (Parallel::Phase::AdjustDomain == next_phase) {
      if constexpr (Parallel::is_dg_element_collection_v<
                        typename metavariables::amr::element_array>) {
        Parallel::threaded_action<Parallel::Actions::SimpleActionOnElement<
            ::amr::Actions::AdjustDomain, true>>(
            Parallel::get_parallel_component<
                typename metavariables::amr::element_array>(local_cache));
      } else {
        Parallel::simple_action<::amr::Actions::AdjustDomain>(
            Parallel::get_parallel_component<
                typename metavariables::amr::element_array>(local_cache));
      }
    }
  }
};
}  // namespace amr
