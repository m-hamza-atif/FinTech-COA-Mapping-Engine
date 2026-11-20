from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

AccountCode = Annotated[str, StringConstraints(min_length=1, max_length=256)]
Description = Annotated[str, StringConstraints(min_length=1, max_length=4096)]
ConfidenceScore = Annotated[float, Field(ge=0.0, le=1.0)]
RoutingStatus = Literal["AUTO_APPROVED", "NEEDS_REVIEW", "EXCLUDED"]


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class Account(ContractModel):
    source_code: AccountCode
    source_name: Description | None
    normal_balance: Description | None
    ending_balance: float | None
    exclude: bool


class AIInput(ContractModel):
    entity_id: AccountCode
    target_schema: AccountCode
    accounts: list[Account] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_source_codes(self):
        if len({account.source_code for account in self.accounts}) != len(
            self.accounts
        ):
            raise ValueError("source_code must be unique within a batch")
        return self


class TopCandidate(ContractModel):
    code: AccountCode
    name: Description
    score: ConfidenceScore


class MappingResult(ContractModel):
    source_code: AccountCode
    source_name: Description | None
    predicted_target_code: AccountCode | None
    predicted_target_name: Description | None
    target_hierarchy: Description | None
    confidence_score: ConfidenceScore
    routing_status: RoutingStatus
    top_k_candidates: list[TopCandidate] = Field(default_factory=list)
    reasoning: Description | None = None


class AIOutput(ContractModel):
    mappings: list[MappingResult]
