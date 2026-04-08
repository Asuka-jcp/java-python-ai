package com.example.gateway.dto;

import jakarta.validation.constraints.NotBlank;

public record HotTopicRequest(
        @NotBlank(message = "platform 不能为空")
        String platform
) {
}
