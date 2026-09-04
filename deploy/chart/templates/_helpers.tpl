{{- define "model-service.name" -}}
{{- required "model is required: a release serves one registered model" .Values.model -}}
{{- end -}}

{{- define "model-service.labels" -}}
app.kubernetes.io/name: {{ include "model-service.name" . }}
app.kubernetes.io/component: model-service
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{- define "model-service.selector" -}}
app.kubernetes.io/name: {{ include "model-service.name" . }}
{{- end -}}
