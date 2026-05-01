from surveyforge.llm.roles import AgentRole


def test_all_roles_defined():
    expected = {
        "PLANNER",
        "RESEARCHER_WIDE",
        "RESEARCHER_DEEP",
        "SYNTHESIZER",
        "WRITER",
        "CRITIC_SECTION",
        "CRITIC_FINAL",
        "JUDGE_DEFAULT",
        "JUDGE_FINAL",
        "LITE_WORKER",
    }
    assert {r.name for r in AgentRole} == expected


def test_role_value_is_lowercase():
    assert AgentRole.PLANNER.value == "planner"
    assert AgentRole.RESEARCHER_WIDE.value == "researcher_wide"
