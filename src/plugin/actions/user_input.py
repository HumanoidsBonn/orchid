"""User-input action for planner-visible clarification."""

from __future__ import annotations

from typing import Any

from src.core.action import ActionResult, AsyncAction
from src.core.context import Context
from src.core.failure import FailureCode, FailureInfo
import src.core.logger as log


class AskUser(AsyncAction):
    """Ask the user a clarification question and then request replanning.

    Use this when the current task cannot be planned safely without information
    from the user, such as choosing between multiple matching objects, confirming
    a destination, or asking for a missing task detail. The action waits
    asynchronously for the session's user-input callback, stores the response in
    ``WorldState``, and returns ``needs_replan`` so the session can ask the LLM
    planner for a revised tree with the new context.

    Params:
        prompt: str - Clear, specific question to present to the user.
        response_key: str - WorldState key used to store the user's response.
                            Defaults to ``last_user_input``.
    """

    def __init__(
        self,
        prompt: str = 'Please clarify the next step.',
        response_key: str = 'last_user_input',
        replan_on_response: bool = True,
        choices: list[dict[str, str]] | None = None,
    ) -> None:
        self.prompt = prompt
        self.response_key = response_key
        self.replan_on_response = replan_on_response
        self.choices = choices
        self._generation = 0
        super().__init__()
        self._status = 'idle'
        self._response: str | None = None
        self._failure_reason: str | None = None

    def start(self, context: Context) -> None:
        self._generation += 1
        generation = self._generation
        self._cancel_requested = False
        self._response = None
        self._failure = None
        if context.world_state is not None:
            context.world_state.set(self.response_key, None)
        log.speak(self.prompt)
        if context.request_user_input is None:
            self._status = "failure"
            self._failure_reason = "User input requested but no callback is configured."
            return

        self._status = "waiting"

        def _on_response(response: str | None) -> None:
            if generation != self._generation or self._cancel_requested or context.execution.cancel_requested or context.execution.control is not None:
                return
            if response == "timeout":
                self._status = "failure"
                self._failure_reason = "User input timed out; no answer received."
                self._failure = FailureInfo(FailureCode.TIMEOUT)
                log.warn(self._failure_reason)
                return
            if response is None or not str(response).strip():
                self._status = "failure"
                self._failure_reason = "No clarification provided by user."
                return
            self._response = str(response).strip()
            if context.world_state is not None:
                context.world_state.set(self.response_key, self._response)
            self._status = "received"
            log.info("User response received")
            if self.replan_on_response:
                log.speak("Thanks. Replanning with your answer.")

        def on_error(exc):
            if generation == self._generation and not self._cancel_requested:
                self._status, self._failure_reason = "failure", str(exc)
                self._failure = FailureInfo(FailureCode.TIMEOUT if isinstance(exc, TimeoutError) else FailureCode.NETWORK_ERROR)
                log.warn("User-input request failed: %s", exc)
        if self.choices is None:
            context.run_in_context_loop(context.request_user_input, self.prompt, callback=_on_response, error_callback=on_error)
        else:
            context.run_in_context_loop(context.request_user_input, self.prompt, choices=list(self.choices), callback=_on_response, error_callback=on_error)

    def check_result(self, context: Context) -> ActionResult:
        if self._cancel_requested:
            return ActionResult(status="failure", reason="Action Cancelled")
        if self._status == "failure":
            return ActionResult(status="failure", reason=self._failure_reason, failure=self._failure)
        if self._status != "received":
            return ActionResult(status="running")

        assert self._response is not None
        if not self.replan_on_response:
            return ActionResult(status="success", payload={"user_input": self._response, "response_key": self.response_key})
        return ActionResult(
            status="needs_replan",
            reason=self.prompt,
            suggested_replan_context={
                "user_input": self._response,
                "response_key": self.response_key,
            },
            payload={"user_input": self._response, "response_key": self.response_key},
        )

    def reset(self) -> None:
        super().reset()
        self._generation += 1
        self._status = "idle"
        self._response = None
        self._failure_reason = None

    def cancel(self, context: Context) -> None:
        self._cancel_requested = True
        self._generation += 1
