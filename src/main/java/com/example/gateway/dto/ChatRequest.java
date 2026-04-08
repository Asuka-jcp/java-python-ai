package com.example.gateway.dto;

import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.Size;

public record ChatRequest(
        @NotBlank(message = "message 不能为空")
        @Size(min = 1, max = 4000, message = "message 长度需在1~4000")
        String message
) {
}
