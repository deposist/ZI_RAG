const state = {
  config: null,
  indexes: [],
  documents: [],
  jobs: [],
  activeIndexId: "",
  activeDocumentId: "",
  selectedDocumentIds: new Set(),
  documentLimit: 100,
  documentOffset: 0,
  documentTotal: 0,
  documentStatusCounts: {},
  documentSearchTimer: null,
  lastDocumentRefresh: 0,
  embeddingModels: [],
  generationModels: [],
  apiKey: window.localStorage.getItem("ziRagApiKey") || "",
  locale: window.localStorage.getItem("ziRagLocale") || "",
  messages: {},
};

const DEFAULT_CONTEXT_TEMPLATE = [
  "Используй контекст RAG ниже как приоритетный источник. Контекст может быть разбит на пачки: просматривай все пачки последовательно и не игнорируй поздние пачки только из-за их номера. Не придумывай локаторы и цитаты. Если ответа в контексте нет, честно скажи, что в базе знаний ответ не найден.",
  "",
  "{knowledge}",
].join("\n");

const DEFAULT_DEEP_TRIGGER_PHRASES = [
  "проанализируй все",
  "проанализировать все",
  "проверь все",
  "проверь всё",
  "проверь все документы",
  "проверь весь пакет",
  "сравни",
  "сравнить",
  "полный перечень",
  "все требования",
  "все нарушения",
  "найди противоречия",
  "найти противоречия",
  "сделай отчет",
  "сделай отчёт",
  "подготовь отчет",
  "подготовь отчёт",
  "ничего не пропусти",
  "полный анализ",
  "по всем документам",
];

const DEFAULT_COMPLIANCE_TRIGGER_PHRASES = [
  "проверь на соответствие",
  "проверка нмд",
  "соответствует ли",
  "найди нарушения",
  "найти нарушения",
  "сделай акт",
  "подготовь акт",
  "проведи проверку",
  "проверить документ",
  "матрица соответствия",
  "compliance",
];

const els = {};

function $(id) {
  return document.getElementById(id);
}

function formatMessage(template, params = {}) {
  return String(template || "").replace(/\{([a-zA-Z0-9_]+)\}/g, (_, key) => {
    return Object.prototype.hasOwnProperty.call(params, key) ? String(params[key]) : `{${key}}`;
  });
}

function t(key, params = {}, fallback = "") {
  return formatMessage(state.messages[key] || fallback || key, params);
}

function browserLocale() {
  const language = String(window.navigator.language || "").toLowerCase();
  return language.startsWith("en") ? "en" : "ru";
}

async function loadMessages(locale = state.locale || browserLocale()) {
  const selected = locale === "en" ? "en" : "ru";
  try {
    const response = await fetch(`/ui/assets/messages.${selected}.json`, {cache: "no-store"});
    if (!response.ok) throw new Error(`messages ${selected}: ${response.status}`);
    state.messages = await response.json();
    state.locale = selected;
  } catch (error) {
    if (selected !== "ru") {
      return loadMessages("ru");
    }
    state.messages = {};
    state.locale = "ru";
  }
  window.localStorage.setItem("ziRagLocale", state.locale);
  document.documentElement.lang = state.locale;
  if (els.languageSelect) {
    els.languageSelect.value = state.locale;
  }
  applyI18n();
}

function applyI18n() {
  document.querySelectorAll("[data-i18n]").forEach((node) => {
    node.textContent = t(node.dataset.i18n, {}, node.textContent);
  });
  document.querySelectorAll("[data-i18n-title]").forEach((node) => {
    node.setAttribute("title", t(node.dataset.i18nTitle, {}, node.getAttribute("title") || ""));
  });
  document.querySelectorAll("[data-i18n-placeholder]").forEach((node) => {
    node.setAttribute("placeholder", t(node.dataset.i18nPlaceholder, {}, node.getAttribute("placeholder") || ""));
  });
}

function showToast(message) {
  els.toast.textContent = message;
  els.toast.classList.remove("is-hidden");
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => els.toast.classList.add("is-hidden"), 4200);
}

function sumGpuMemory(items, key) {
  return (items || []).reduce((total, item) => total + Number(item?.[key] || 0), 0);
}

function modelName(item) {
  return String(item?.name || item?.model || item?.id || "").trim();
}

function uniqueModelNames(models) {
  return [...new Set((models || []).map(modelName).filter(Boolean))]
    .sort((left, right) => left.localeCompare(right));
}

function likelyGenerationModel(item) {
  const name = modelName(item).toLowerCase();
  const details = item?.details || {};
  const families = [details.family, ...(details.families || [])]
    .filter(Boolean)
    .map((family) => String(family).toLowerCase());
  if (families.includes("bert")) return false;
  if (name.includes("bge") || name.includes("embedding")) return false;
  return true;
}

function renderModelSelect(select, models, selected, placeholder) {
  if (!select) return;
  const current = String(selected || "").trim();
  const names = uniqueModelNames(models);
  const hasCurrent = current && names.includes(current);
  const options = [`<option value="">${escapeHtml(placeholder)}</option>`];
  if (current && !hasCurrent) {
    options.push(
      `<option value="${escapeHtml(current)}" selected>${escapeHtml(current)} · ${escapeHtml(t("models.current", {}, "текущее"))}</option>`
    );
  }
  options.push(...names.map((name) => {
    const selectedAttr = name === current ? " selected" : "";
    return `<option value="${escapeHtml(name)}"${selectedAttr}>${escapeHtml(name)}</option>`;
  }));
  select.innerHTML = options.join("");
  select.value = current;
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: {
      ...(options.body instanceof FormData ? {} : {"Content-Type": "application/json"}),
      ...(state.apiKey ? {"X-API-Key": state.apiKey} : {}),
      ...(options.headers || {}),
    },
  });
  const text = await response.text();
  const data = text ? JSON.parse(text) : {};
  if (!response.ok) {
    if (response.status === 401) {
      const key = window.prompt(t("api.keyPrompt", {}, "Введите API key для RAG sidecar"), state.apiKey);
      if (key !== null) {
        state.apiKey = key.trim();
        window.localStorage.setItem("ziRagApiKey", state.apiKey);
        return api(path, options);
      }
    }
    throw new Error(data.detail || data.error || response.statusText);
  }
  return data;
}

function statusPill(status) {
  return `<span class="status ${String(status || "").toLowerCase()}">${status || "unknown"}</span>`;
}

function filteredDocuments() {
  return state.documents;
}

function isUnindexedDocument(item) {
  return String(item?.status || "").toLowerCase() !== "indexed";
}

function renderIndexes() {
  els.deleteIndexBtn.disabled = !state.activeIndexId;
  els.indexList.innerHTML = state.indexes.map((item) => {
    const active = item.id === state.activeIndexId ? " is-active" : "";
    return `
      <button class="index-item${active}" type="button" data-index-id="${item.id}">
        <span class="item-title">
          <span>${escapeHtml(item.name || item.id)}</span>
          ${statusPill(item.status)}
        </span>
        <span class="item-meta">${Number(item.document_count || 0)} docs · ${Number(item.chunk_count || 0)} chunks</span>
      </button>
    `;
  }).join("");
}

function renderDocuments() {
  const docs = filteredDocuments();
  els.documentList.innerHTML = docs.map((item) => {
    const active = item.id === state.activeDocumentId ? " is-active" : "";
    const selected = state.selectedDocumentIds.has(item.id);
    const checked = selected ? " checked" : "";
    const selectedClass = selected ? " is-selected" : "";
    return `
      <div class="doc-item${active}${selectedClass}">
        <label class="doc-check" title="${escapeHtml(t("documents.selectTitle", {}, "Выбрать документ"))}">
          <input class="doc-checkbox" type="checkbox" data-document-id="${item.id}"${checked}>
        </label>
        <button class="doc-main" type="button" data-document-id="${item.id}">
          <span class="item-title">
            <span>${escapeHtml(item.filename || item.id)}</span>
            ${statusPill(item.status)}
          </span>
          <span class="item-meta">${Number(item.chunk_count || 0)} chunks · ${escapeHtml(item.source_path || item.stored_path || "")}</span>
        </button>
      </div>
    `;
  }).join("") || `<div class="empty" style="padding:14px">${escapeHtml(t("documents.empty", {}, "Документов пока нет."))}</div>`;
  renderBulkControls(docs);
}

function renderDetails() {
  const doc = state.documents.find((item) => item.id === state.activeDocumentId);
  els.detailsEmpty.classList.toggle("is-hidden", Boolean(doc));
  els.detailsContent.classList.toggle("is-hidden", !doc);
  if (!doc) return;
  els.detailName.textContent = doc.filename || doc.id;
  els.detailStatus.innerHTML = statusPill(doc.status);
  els.detailChunks.textContent = String(doc.chunk_count || 0);
  els.detailHash.textContent = doc.file_hash || "";
  els.detailSource.textContent = doc.source_path || doc.stored_path || "";
  els.detailError.textContent = doc.error || "";
}

function renderJobs() {
  const activeJobs = state.jobs.filter((item) => ["queued", "running", "cancel_requested"].includes(item.status));
  const cancelableJobs = activeJobs.filter((item) => item.status !== "cancel_requested");
  els.stopJobsBtn.disabled = !state.activeIndexId || cancelableJobs.length === 0;
  els.stopJobsBtn.textContent = cancelableJobs.length
    ? t("jobs.stopCount", {count: cancelableJobs.length}, "Остановить ({count})")
    : t("jobs.stop", {}, "Остановить");
  if (!state.activeIndexId) {
    els.jobsLine.textContent = t("jobs.chooseIndex", {}, "Выберите индекс");
  } else if (activeJobs.some((item) => item.status === "cancel_requested")) {
    els.jobsLine.textContent = t("jobs.stopping", {count: activeJobs.length}, "Остановка индексирования: {count} задач");
  } else if (activeJobs.length) {
    const currentJob = activeJobs.find((item) => item.status === "running") || activeJobs[0];
    const message = currentJob?.message ? ` · ${currentJob.message}` : "";
    els.jobsLine.textContent = t("jobs.indexing", {count: activeJobs.length, message}, "Индексирование: {count} задач{message}");
  } else {
    els.jobsLine.textContent = t("jobs.idle", {}, "Индексирование не запущено");
  }
}

function renderBulkControls(docs = filteredDocuments()) {
  const visibleIds = docs.map((item) => item.id);
  const statusCounts = state.documentStatusCounts || {};
  const indexedCount = Number(statusCounts.indexed || 0);
  const totalCount = Object.values(statusCounts).reduce((sum, value) => sum + Number(value || 0), 0);
  const unindexedCount = Math.max(0, totalCount - indexedCount);
  const selectedCount = state.selectedDocumentIds.size;
  const visibleSelectedCount = visibleIds.filter((id) => state.selectedDocumentIds.has(id)).length;
  const start = state.documentTotal ? state.documentOffset + 1 : 0;
  const end = Math.min(state.documentOffset + docs.length, state.documentTotal);
  els.selectedDocsLine.textContent = t("documents.selected", {count: selectedCount}, "Выбрано: {count}");
  els.bulkReindexBtn.disabled = !state.activeIndexId || selectedCount === 0;
  els.bulkForceReindexBtn.disabled = !state.activeIndexId || selectedCount === 0;
  els.bulkDeleteBtn.disabled = !state.activeIndexId || selectedCount === 0;
  els.clearSelectionBtn.disabled = selectedCount === 0;
  els.selectUnindexedBtn.disabled = !state.activeIndexId || unindexedCount === 0;
  els.selectUnindexedBtn.textContent = unindexedCount
    ? t("documents.unindexedCount", {count: unindexedCount}, "Неиндексированные ({count})")
    : t("documents.unindexed", {}, "Неиндексированные");
  els.selectVisibleInput.disabled = !state.activeIndexId || visibleIds.length === 0;
  els.selectVisibleInput.checked = visibleIds.length > 0 && visibleSelectedCount === visibleIds.length;
  els.selectVisibleInput.indeterminate = visibleSelectedCount > 0 && visibleSelectedCount < visibleIds.length;
  els.documentPageLine.textContent = t("documents.page", {start, end, total: state.documentTotal}, "{start}–{end} из {total}");
  els.prevDocsBtn.disabled = !state.activeIndexId || state.documentOffset <= 0;
  els.nextDocsBtn.disabled = !state.activeIndexId || state.documentOffset + state.documentLimit >= state.documentTotal;
}

function renderDefaultIndexOptions() {
  const selected = (state.config?.default_index_ids || [])[0] || "";
  els.defaultIndexSelect.innerHTML = `<option value="">${escapeHtml(t("indexes.all", {}, "Все индексы"))}</option>` + state.indexes.map((item) => {
    const label = item.name && item.name !== item.id ? `${item.name} (${item.id})` : item.id;
    return `<option value="${escapeHtml(item.id)}">${escapeHtml(label)}</option>`;
  }).join("");
  els.defaultIndexSelect.value = selected;
  if (els.complianceIndexSelect) {
    const complianceSelected = (state.config?.compliance_index_ids || [])[0] || "";
    els.complianceIndexSelect.innerHTML = `<option value="">${escapeHtml(t("indexes.default", {}, "Как индекс по умолчанию"))}</option>` + state.indexes.map((item) => {
      const label = item.name && item.name !== item.id ? `${item.name} (${item.id})` : item.id;
      return `<option value="${escapeHtml(item.id)}">${escapeHtml(label)}</option>`;
    }).join("");
    els.complianceIndexSelect.value = complianceSelected;
  }
}

async function loadConfig() {
  state.config = await api("/config");
  els.ollamaUrlInput.value = state.config.ollama_base_url || "";
  els.apiKeyInput.value = state.apiKey || state.config.api_key || "";
  els.requireApiKeyLocalhostInput.checked = Boolean(state.config.require_api_key_localhost);
  els.embeddingProviderInput.value = state.config.embedding_provider || "ollama";
  els.embeddingModelInput.value = state.config.embedding_model || "";
  els.embeddingBaseUrlInput.value = state.config.embedding_base_url || "";
  els.embeddingApiKeyInput.value = state.config.embedding_api_key || "";
  els.embeddingBatchSizeInput.value = state.config.embedding_batch_size || 16;
  els.embeddingCacheDtypeInput.value = state.config.embedding_cache_dtype || "fp32";
  els.embeddingQueryPrefixInput.value = state.config.embedding_query_prefix || "";
  els.embeddingDocumentPrefixInput.value = state.config.embedding_document_prefix || "";
  els.storageDirInput.value = state.config.storage_dir || "";
  els.allowedRootsInput.value = (state.config.allowed_source_roots || []).join("\n");
  els.chunkSizeInput.value = state.config.chunk_size || 1200;
  els.chunkOverlapInput.value = state.config.chunk_overlap || 120;
  els.indexTypeInput.value = state.config.index_type || "auto";
  els.hnswThresholdInput.value = state.config.hnsw_threshold_chunks || 50000;
  els.hnswMInput.value = state.config.hnsw_m || 32;
  els.hnswEfConstructionInput.value = state.config.hnsw_ef_construction || 200;
  els.hnswEfSearchInput.value = state.config.hnsw_ef_search || 128;
  els.topKInput.value = state.config.top_k || 8;
  els.scoreInput.value = state.config.score_threshold ?? 0.50;
  els.ragEnabledInput.checked = state.config.rag_enabled !== false;
  els.includeSourcesInput.checked = state.config.include_sources !== false;
  els.deepAnalysisEnabledInput.checked = state.config.deep_analysis_enabled !== false;
  els.deepFinalAnswerInput.checked = state.config.deep_final_answer !== false;
  els.deepForceAllInput.checked = Boolean(state.config.deep_force_all);
  els.retrievalTopKInput.value = state.config.retrieval_top_k || 70;
  els.maxPromptChunksInput.value = state.config.max_prompt_chunks || 24;
  els.adaptiveScoreMarginInput.value = state.config.adaptive_score_margin ?? 0.20;
  els.minQueryTermHitsInput.value = state.config.min_query_term_hits ?? 1;
  els.queryExpansionEnabledInput.checked = Boolean(state.config.query_expansion_enabled);
  els.queryExpansionModelInput.value = state.config.query_expansion_model || "";
  els.queryExpansionMaxVariantsInput.value = state.config.query_expansion_max_variants || 3;
  els.queryExpansionMaxTokensInput.value = state.config.query_expansion_max_tokens || 256;
  els.rerankEnabledInput.checked = Boolean(state.config.rerank_enabled);
  els.rerankModelInput.value = state.config.rerank_model || "";
  els.rerankMinResultsInput.value = state.config.rerank_min_results || 10;
  els.rerankTopNInput.value = state.config.rerank_top_n || 50;
  els.maxContextCharsInput.value = state.config.max_context_chars || 32000;
  els.contextBatchCharsInput.value = state.config.context_batch_chars || 10000;
  els.maxContextBatchesInput.value = state.config.max_context_batches || 3;
  els.maxCompactSourcesInput.value = state.config.max_compact_sources ?? 8;
  els.contextTemplateInput.value = state.config.context_template || DEFAULT_CONTEXT_TEMPLATE;
  els.chatAttachmentsEnabledInput.checked = state.config.chat_attachments_enabled !== false;
  els.chatAttachmentIndexPrefixInput.value = state.config.chat_attachment_index_prefix || "owui_chat_";
  els.chatAttachmentMaxFilesInput.value = state.config.chat_attachment_max_files || 10;
  els.chatAttachmentMaxFileMbInput.value = state.config.chat_attachment_max_file_mb || 256;
  els.chatAttachmentTimeoutInput.value = state.config.chat_attachment_timeout_sec || 900;
  els.deepGenerationProviderInput.value = state.config.deep_generation_provider || "ollama";
  els.deepGenerationBaseUrlInput.value = state.config.deep_generation_base_url || "";
  els.deepGenerationApiKeyInput.value = state.config.deep_generation_api_key || "";
  els.deepGenerationModelInput.value = state.config.deep_generation_model || "";
  els.deepTopKInput.value = state.config.deep_top_k || 70;
  els.deepMaxBatchesInput.value = state.config.deep_max_batches || 10;
  els.deepBatchCharsInput.value = state.config.deep_batch_chars || 10000;
  els.deepBatchTokensInput.value = state.config.deep_batch_max_tokens || 1024;
  els.deepFinalTokensInput.value = state.config.deep_final_max_tokens || 2048;
  els.deepTimeoutInput.value = state.config.deep_timeout_sec || 900;
  els.deepTriggerPhrasesInput.value = (state.config.deep_trigger_phrases || DEFAULT_DEEP_TRIGGER_PHRASES).join("\n");
  els.complianceEnabledInput.checked = state.config.compliance_enabled !== false;
  els.complianceAutoInput.checked = state.config.compliance_auto_enabled !== false;
  els.complianceAllowUserIndexInput.checked = state.config.compliance_allow_user_index_override !== false;
  els.complianceGenerationModelInput.value = state.config.compliance_generation_model || "";
  els.complianceMaxFilesInput.value = state.config.compliance_max_files || 10;
  els.complianceMaxFileMbInput.value = state.config.compliance_max_file_mb || 256;
  els.complianceSectionCharsInput.value = state.config.compliance_section_chars || 8000;
  els.complianceMaxSectionsInput.value = state.config.compliance_max_sections || 80;
  els.complianceTopKInput.value = state.config.compliance_requirement_top_k || 24;
  els.complianceTimeoutInput.value = state.config.compliance_timeout_sec || 1200;
  els.complianceTriggerPhrasesInput.value = (state.config.compliance_trigger_phrases || DEFAULT_COMPLIANCE_TRIGGER_PHRASES).join("\n");
  els.ocrInput.checked = Boolean(state.config.enable_ocr);
  els.ocrEngineInput.value = state.config.ocr_engine || "easyocr";
  els.ocrGpuInput.checked = state.config.ocr_gpu !== false;
  els.ocrGpuDeviceInput.value = state.config.ocr_gpu_device || "";
  els.ocrModelStorageInput.value = state.config.ocr_model_storage_dir || "";
  els.ocrLanguagesInput.value = state.config.ocr_languages || "rus+eng";
  els.pdfRenderScaleInput.value = state.config.pdf_render_scale || 2.5;
  renderDefaultIndexOptions();
}

async function loadHealth() {
  const response = await fetch("/health", {
    headers: {
      ...(state.apiKey ? {"X-API-Key": state.apiKey} : {}),
    },
  });
  const text = await response.text();
  const data = text ? JSON.parse(text) : {};
  const label = data.status === "ok" ? "online" : "degraded";
  els.healthLine.textContent = `Sidecar ${label} · ${data.storage_dir || ""}`;
}

async function loadModels() {
  const selected = state.config?.embedding_model || "";
  try {
    const data = await api("/embedding/models");
    state.embeddingModels = data.models || [];
    renderModelSelect(els.embeddingModelSelect, state.embeddingModels, selected, "Embedding model");
    renderModelSelect(els.embeddingModelInput, state.embeddingModels, selected, t("models.embeddingChoose", {}, "Выберите embedding model"));
  } catch (error) {
    state.embeddingModels = [];
    renderModelSelect(els.embeddingModelSelect, [], selected, t("models.embeddingUnavailable", {}, "Embedding model недоступна"));
    renderModelSelect(els.embeddingModelInput, [], selected, t("models.embeddingUnavailable", {}, "Embedding model недоступна"));
  }
}

async function loadGenerationModels() {
  const deepSelected = state.config?.deep_generation_model || "";
  const complianceSelected = state.config?.compliance_generation_model || "";
  const queryExpansionSelected = state.config?.query_expansion_model || "";
  try {
    const data = await api("/ollama/models");
    const models = data.models || [];
    const generationModels = models.filter(likelyGenerationModel);
    state.generationModels = generationModels.length ? generationModels : models;
    const deepPlaceholder = state.generationModels.length
      ? t("models.notSelected", {}, "Модель не выбрана")
      : t("models.empty", {}, "Список моделей пуст");
    const compliancePlaceholder = state.generationModels.length
      ? t("models.notSelectedDeep", {}, "Модель не выбрана (Deep RAG)")
      : t("models.empty", {}, "Список моделей пуст");
    const queryExpansionPlaceholder = state.generationModels.length
      ? t("models.notSelected", {}, "Модель не выбрана")
      : t("models.empty", {}, "Список моделей пуст");
    renderModelSelect(els.queryExpansionModelInput, state.generationModels, queryExpansionSelected, queryExpansionPlaceholder);
    renderModelSelect(els.deepGenerationModelInput, state.generationModels, deepSelected, deepPlaceholder);
    renderModelSelect(els.complianceGenerationModelInput, state.generationModels, complianceSelected, compliancePlaceholder);
  } catch (error) {
    state.generationModels = [];
    renderModelSelect(els.queryExpansionModelInput, [], queryExpansionSelected, t("models.ollamaError", {}, "Ошибка /ollama/models"));
    renderModelSelect(els.deepGenerationModelInput, [], deepSelected, t("models.ollamaError", {}, "Ошибка /ollama/models"));
    renderModelSelect(els.complianceGenerationModelInput, [], complianceSelected, t("models.ollamaError", {}, "Ошибка /ollama/models"));
  }
}

async function loadIndexes() {
  const data = await api("/indexes");
  state.indexes = data.indexes || [];
  if (state.activeIndexId && !state.indexes.some((item) => item.id === state.activeIndexId)) {
    state.activeIndexId = "";
  }
  if (!state.activeIndexId && state.indexes.length) {
    state.activeIndexId = state.indexes[0].id;
  }
  renderIndexes();
  renderDefaultIndexOptions();
}

async function loadDocuments() {
  if (!state.activeIndexId) {
    state.documents = [];
    state.documentTotal = 0;
    state.documentOffset = 0;
    state.documentStatusCounts = {};
    state.jobs = [];
    state.selectedDocumentIds.clear();
    els.documentsTitle.textContent = t("documents.title", {}, "Документы");
    els.documentsHint.textContent = t("documents.chooseIndex", {}, "Выберите индекс");
    renderDocuments();
    renderDetails();
    renderJobs();
    return;
  }
  const index = state.indexes.find((item) => item.id === state.activeIndexId);
  els.documentsTitle.textContent = index?.name || state.activeIndexId;
  const model = index?.embedding_model || state.config?.embedding_model || t("models.embeddingMissing", {}, "embedding model не задан");
  els.documentsHint.textContent = `${model} · ${index?.embedding_dim || 0} dim`;
  const params = new URLSearchParams({
    limit: String(state.documentLimit),
    offset: String(state.documentOffset),
  });
  const query = (els.documentSearch.value || "").trim();
  const status = els.documentStatusFilter.value || "";
  if (query) params.set("query", query);
  if (status) params.set("status", status);
  const data = await api(`/indexes/${encodeURIComponent(state.activeIndexId)}/documents?${params.toString()}`);
  state.documents = data.documents || [];
  state.documentTotal = Number(data.total || state.documents.length);
  state.documentOffset = Number(data.offset || 0);
  state.documentLimit = Number(data.limit || state.documentLimit);
  state.documentStatusCounts = data.status_counts || {};
  if (!state.documents.length && state.documentTotal > 0 && state.documentOffset >= state.documentTotal) {
    state.documentOffset = Math.max(0, state.documentTotal - state.documentLimit);
    return loadDocuments();
  }
  state.lastDocumentRefresh = Date.now();
  if (!state.documents.some((item) => item.id === state.activeDocumentId)) {
    state.activeDocumentId = "";
  }
  renderDocuments();
  renderDetails();
}

function scheduleDocumentReload() {
  window.clearTimeout(state.documentSearchTimer);
  state.documentOffset = 0;
  state.documentSearchTimer = window.setTimeout(() => {
    loadDocuments().catch((error) => showToast(error.message));
  }, 250);
}

async function changeDocumentPage(direction) {
  const nextOffset = state.documentOffset + direction * state.documentLimit;
  state.documentOffset = Math.max(0, Math.min(nextOffset, Math.max(0, state.documentTotal - 1)));
  await loadDocuments();
}

async function loadJobs() {
  if (!state.activeIndexId) {
    state.jobs = [];
    renderJobs();
    return;
  }
  const data = await api(`/jobs?index_id=${encodeURIComponent(state.activeIndexId)}&active=true`);
  state.jobs = data.jobs || [];
  renderJobs();
}

async function refreshAll() {
  try {
    await Promise.all([loadHealth(), loadConfig()]);
    await Promise.all([loadModels(), loadGenerationModels(), loadIndexes()]);
    await Promise.all([loadDocuments(), loadJobs()]);
  } catch (error) {
    showToast(error.message || t("toast.loadError", {}, "Ошибка загрузки"));
  }
}

async function createIndex(event) {
  event.preventDefault();
  const name = els.newIndexName.value.trim();
  if (!name) return;
  try {
    const body = {
      name,
      embedding_model: state.config?.embedding_model || els.embeddingModelSelect.value || "",
    };
    const created = await api("/indexes", {method: "POST", body: JSON.stringify(body)});
    state.activeIndexId = created.id;
    els.newIndexName.value = "";
    els.newIndexForm.classList.add("is-hidden");
    await loadIndexes();
    await loadDocuments();
  } catch (error) {
    showToast(error.message);
  }
}

async function deleteActiveIndex() {
  if (!state.activeIndexId) return;
  const index = state.indexes.find((item) => item.id === state.activeIndexId);
  const label = index?.name || state.activeIndexId;
  if (!window.confirm(t("confirm.deleteIndex", {label}, 'Удалить индекс "{label}" и все его документы?'))) return;
  try {
    await api(`/indexes/${encodeURIComponent(state.activeIndexId)}`, {method: "DELETE"});
    state.activeIndexId = "";
    state.activeDocumentId = "";
    state.selectedDocumentIds.clear();
    state.documents = [];
    state.jobs = [];
    await loadConfig();
    await loadIndexes();
    await loadDocuments();
    await loadJobs();
    showToast(t("toast.indexDeleted", {}, "Индекс удалён"));
  } catch (error) {
    showToast(error.message);
  }
}

async function uploadFiles() {
  if (!state.activeIndexId || !els.uploadInput.files.length) return;
  const files = [...els.uploadInput.files];
  const form = new FormData();
  for (const file of files) {
    form.append("files", file);
  }
  try {
    const data = await api(`/indexes/${encodeURIComponent(state.activeIndexId)}/documents/upload-batch`, {
      method: "POST",
      body: form,
    });
    showToast(t("toast.uploadQueued", {count: (data.documents || []).length || files.length}, "Загрузка поставлена в очередь: {count}"));
  } catch (error) {
    showToast(error.message);
  }
  els.uploadInput.value = "";
  await loadDocuments();
  await loadJobs();
}

async function addPath() {
  if (!state.activeIndexId) return;
  const path = els.pathInput.value.trim();
  if (!path) return;
  try {
    const data = await api(`/indexes/${encodeURIComponent(state.activeIndexId)}/documents/add-path`, {
      method: "POST",
      body: JSON.stringify({
        path,
        recursive: els.recursiveInput.checked,
        index_now: true,
      }),
    });
    showToast(t("toast.documentsAdded", {count: (data.documents || []).length}, "Добавлено документов: {count}"));
    els.pathInput.value = "";
    els.pathForm.classList.add("is-hidden");
    await loadDocuments();
    await loadJobs();
  } catch (error) {
    showToast(error.message);
  }
}

async function reindexSelected() {
  if (!state.activeIndexId || !state.activeDocumentId) return;
  try {
    await api(`/indexes/${encodeURIComponent(state.activeIndexId)}/documents/${encodeURIComponent(state.activeDocumentId)}/reindex`, {method: "POST"});
    showToast(t("toast.reindexQueued", {}, "Переиндексация поставлена в очередь"));
    await loadDocuments();
    await loadJobs();
  } catch (error) {
    showToast(error.message);
  }
}

async function bulkReindexSelected(force = false) {
  if (!state.activeIndexId || state.selectedDocumentIds.size === 0) return;
  const documentIds = [...state.selectedDocumentIds];
  if (force && !window.confirm(t(
    "confirm.forceReindex",
    {count: documentIds.length},
    "Принудительно переиндексировать выбранные документы: {count}? Активные задачи по ним будут отменены.",
  ))) {
    return;
  }
  try {
    const data = await api(`/indexes/${encodeURIComponent(state.activeIndexId)}/documents/reindex`, {
      method: "POST",
      body: JSON.stringify({
        document_ids: documentIds,
        force,
      }),
    });
    const jobs = data.jobs || [];
    const skipped = data.skipped || [];
    const cancelled = data.cancelled || [];
    const queued = data.document_count ?? jobs.length;
    const parts = [t("toast.queued", {count: queued}, "Поставлено в очередь: {count}")];
    if (skipped.length) parts.push(t("toast.skipped", {count: skipped.length}, "пропущено: {count}"));
    if (cancelled.length) parts.push(t("toast.cancelledJobs", {count: cancelled.length}, "отменено задач: {count}"));
    showToast(parts.join(", "));
    await loadDocuments();
    await loadJobs();
  } catch (error) {
    showToast(error.message);
  }
}

async function bulkDeleteSelected() {
  if (!state.activeIndexId || state.selectedDocumentIds.size === 0) return;
  const documentIds = [...state.selectedDocumentIds];
  const index = state.indexes.find((item) => item.id === state.activeIndexId);
  const label = index?.name || state.activeIndexId;
  if (!window.confirm(t("confirm.deleteDocuments", {label, count: documentIds.length}, 'Удалить выбранные документы из индекса "{label}": {count}?'))) {
    return;
  }
  try {
    const data = await api(`/indexes/${encodeURIComponent(state.activeIndexId)}/documents/delete`, {
      method: "POST",
      body: JSON.stringify({document_ids: documentIds}),
    });
    const deleted = data.deleted || [];
    const missing = data.missing || [];
    const cancelled = data.cancelled || [];
    const deletedIds = new Set(deleted.map((item) => item.id));
    for (const documentId of deletedIds) {
      state.selectedDocumentIds.delete(documentId);
    }
    if (deletedIds.has(state.activeDocumentId)) {
      state.activeDocumentId = "";
    }
    const parts = [t("toast.deleted", {count: data.deleted_count ?? deleted.length}, "Удалено: {count}")];
    if (missing.length) parts.push(t("toast.missing", {count: missing.length}, "не найдено: {count}"));
    if (cancelled.length) parts.push(t("toast.stoppedJobs", {count: cancelled.length}, "остановлено задач: {count}"));
    showToast(parts.join(", "));
    await loadIndexes();
    await loadDocuments();
    await loadJobs();
  } catch (error) {
    showToast(error.message);
  }
}

function toggleVisibleSelection() {
  const docs = filteredDocuments();
  if (!docs.length) return;
  const allSelected = docs.every((item) => state.selectedDocumentIds.has(item.id));
  for (const item of docs) {
    if (allSelected) {
      state.selectedDocumentIds.delete(item.id);
    } else {
      state.selectedDocumentIds.add(item.id);
    }
  }
  renderDocuments();
}

function selectUnindexedDocuments() {
  selectUnindexedDocumentsAsync().catch((error) => showToast(error.message));
}

async function selectUnindexedDocumentsAsync() {
  if (!state.activeIndexId) return;
  const params = new URLSearchParams({
    limit: "500",
    offset: "0",
    status: "unindexed",
  });
  const data = await api(`/indexes/${encodeURIComponent(state.activeIndexId)}/documents?${params.toString()}`);
  const docs = data.documents || [];
  const total = Number(data.total || docs.length);
  state.selectedDocumentIds.clear();
  for (const item of docs) {
    state.selectedDocumentIds.add(item.id);
  }
  renderDocuments();
  if (total > docs.length) {
    showToast(t("toast.unindexedSelectedLimited", {count: docs.length, total}, "Выбрано неиндексированных: {count} из {total}; максимум за раз 500"));
  } else {
    showToast(docs.length
      ? t("toast.unindexedSelected", {count: docs.length}, "Выбрано неиндексированных: {count}")
      : t("toast.noUnindexed", {}, "Неиндексированных документов нет"));
  }
}

function clearDocumentSelection() {
  state.selectedDocumentIds.clear();
  renderDocuments();
}

async function stopIndexing() {
  if (!state.activeIndexId || els.stopJobsBtn.disabled) return;
  try {
    const data = await api(`/indexes/${encodeURIComponent(state.activeIndexId)}/jobs/cancel`, {method: "POST"});
    showToast(t("toast.stopRequested", {count: (data.jobs || []).length}, "Остановка запрошена: {count} задач"));
    await loadJobs();
    await loadDocuments();
  } catch (error) {
    showToast(error.message);
  }
}

async function deleteSelected() {
  if (!state.activeIndexId || !state.activeDocumentId) return;
  if (!window.confirm(t("confirm.deleteDocument", {}, "Удалить документ из индекса?"))) return;
  try {
    await api(`/indexes/${encodeURIComponent(state.activeIndexId)}/documents/${encodeURIComponent(state.activeDocumentId)}`, {method: "DELETE"});
    state.selectedDocumentIds.delete(state.activeDocumentId);
    state.activeDocumentId = "";
    await loadDocuments();
    showToast(t("toast.documentDeleted", {}, "Документ удалён"));
  } catch (error) {
    showToast(error.message);
  }
}

async function saveSettings() {
  const payload = {
    ollama_base_url: els.ollamaUrlInput.value.trim(),
    api_key: els.apiKeyInput.value.trim(),
    require_api_key_localhost: els.requireApiKeyLocalhostInput.checked,
    embedding_provider: els.embeddingProviderInput.value || "ollama",
    embedding_model: els.embeddingModelInput.value.trim() || els.embeddingModelSelect.value,
    embedding_base_url: els.embeddingBaseUrlInput.value.trim(),
    embedding_api_key: els.embeddingApiKeyInput.value.trim(),
    embedding_batch_size: Number(els.embeddingBatchSizeInput.value || 16),
    embedding_cache_dtype: els.embeddingCacheDtypeInput.value || "fp32",
    embedding_query_prefix: els.embeddingQueryPrefixInput.value,
    embedding_document_prefix: els.embeddingDocumentPrefixInput.value,
    default_index_ids: els.defaultIndexSelect.value ? [els.defaultIndexSelect.value] : [],
    storage_dir: els.storageDirInput.value.trim(),
    allowed_source_roots: els.allowedRootsInput.value.split("\n").map((line) => line.trim()).filter(Boolean),
    chunk_size: Number(els.chunkSizeInput.value || 1200),
    chunk_overlap: Number(els.chunkOverlapInput.value || 120),
    index_type: els.indexTypeInput.value || "auto",
    hnsw_threshold_chunks: Number(els.hnswThresholdInput.value || 50000),
    hnsw_m: Number(els.hnswMInput.value || 32),
    hnsw_ef_construction: Number(els.hnswEfConstructionInput.value || 200),
    hnsw_ef_search: Number(els.hnswEfSearchInput.value || 128),
    top_k: Number(els.topKInput.value || 8),
    score_threshold: Number(els.scoreInput.value || 0.50),
    rag_enabled: els.ragEnabledInput.checked,
    include_sources: els.includeSourcesInput.checked,
    retrieval_top_k: Number(els.retrievalTopKInput.value || 70),
    adaptive_score_margin: Number(els.adaptiveScoreMarginInput.value || 0.20),
    max_prompt_chunks: Number(els.maxPromptChunksInput.value || 24),
    min_query_term_hits: Number(els.minQueryTermHitsInput.value || 1),
    query_expansion_enabled: els.queryExpansionEnabledInput.checked,
    query_expansion_model: els.queryExpansionModelInput.value.trim(),
    query_expansion_max_variants: Number(els.queryExpansionMaxVariantsInput.value || 3),
    query_expansion_max_tokens: Number(els.queryExpansionMaxTokensInput.value || 256),
    rerank_enabled: els.rerankEnabledInput.checked,
    rerank_model: els.rerankModelInput.value.trim(),
    rerank_min_results: Number(els.rerankMinResultsInput.value || 10),
    rerank_top_n: Number(els.rerankTopNInput.value || 50),
    max_context_chars: Number(els.maxContextCharsInput.value || 32000),
    context_batch_chars: Number(els.contextBatchCharsInput.value || 10000),
    max_context_batches: Number(els.maxContextBatchesInput.value || 3),
    max_compact_sources: Number(els.maxCompactSourcesInput.value || 8),
    context_template: els.contextTemplateInput.value.trim() || DEFAULT_CONTEXT_TEMPLATE,
    chat_attachments_enabled: els.chatAttachmentsEnabledInput.checked,
    chat_attachment_index_prefix: els.chatAttachmentIndexPrefixInput.value.trim() || "owui_chat_",
    chat_attachment_max_files: Number(els.chatAttachmentMaxFilesInput.value || 10),
    chat_attachment_max_file_mb: Number(els.chatAttachmentMaxFileMbInput.value || 256),
    chat_attachment_timeout_sec: Number(els.chatAttachmentTimeoutInput.value || 900),
    deep_analysis_enabled: els.deepAnalysisEnabledInput.checked,
    deep_final_answer: els.deepFinalAnswerInput.checked,
    deep_force_all: els.deepForceAllInput.checked,
    deep_trigger_phrases: els.deepTriggerPhrasesInput.value.split("\n").map((line) => line.trim()).filter(Boolean),
    deep_generation_provider: els.deepGenerationProviderInput.value || "ollama",
    deep_generation_base_url: els.deepGenerationBaseUrlInput.value.trim(),
    deep_generation_api_key: els.deepGenerationApiKeyInput.value.trim(),
    deep_generation_model: els.deepGenerationModelInput.value.trim(),
    deep_top_k: Number(els.deepTopKInput.value || 70),
    deep_max_batches: Number(els.deepMaxBatchesInput.value || 10),
    deep_batch_chars: Number(els.deepBatchCharsInput.value || 10000),
    deep_batch_max_tokens: Number(els.deepBatchTokensInput.value || 1024),
    deep_final_max_tokens: Number(els.deepFinalTokensInput.value || 2048),
    deep_timeout_sec: Number(els.deepTimeoutInput.value || 900),
    compliance_enabled: els.complianceEnabledInput.checked,
    compliance_auto_enabled: els.complianceAutoInput.checked,
    compliance_allow_user_index_override: els.complianceAllowUserIndexInput.checked,
    compliance_index_ids: els.complianceIndexSelect.value ? [els.complianceIndexSelect.value] : [],
    compliance_generation_model: els.complianceGenerationModelInput.value.trim(),
    compliance_max_files: Number(els.complianceMaxFilesInput.value || 10),
    compliance_max_file_mb: Number(els.complianceMaxFileMbInput.value || 256),
    compliance_section_chars: Number(els.complianceSectionCharsInput.value || 8000),
    compliance_max_sections: Number(els.complianceMaxSectionsInput.value || 80),
    compliance_requirement_top_k: Number(els.complianceTopKInput.value || 24),
    compliance_timeout_sec: Number(els.complianceTimeoutInput.value || 1200),
    compliance_trigger_phrases: els.complianceTriggerPhrasesInput.value.split("\n").map((line) => line.trim()).filter(Boolean),
    enable_ocr: els.ocrInput.checked,
    ocr_engine: els.ocrEngineInput.value || "easyocr",
    ocr_gpu: els.ocrGpuInput.checked,
    ocr_gpu_device: els.ocrGpuDeviceInput.value.trim(),
    ocr_model_storage_dir: els.ocrModelStorageInput.value.trim(),
    ocr_languages: els.ocrLanguagesInput.value.trim() || "rus+eng",
    pdf_render_scale: Number(els.pdfRenderScaleInput.value || 2.5),
  };
  try {
    await api("/config", {method: "PUT", body: JSON.stringify(payload)});
    state.apiKey = payload.api_key || "";
    window.localStorage.setItem("ziRagApiKey", state.apiKey);
    els.settingsDialog.close();
    await refreshAll();
    showToast(t("toast.settingsSaved", {}, "Настройки сохранены"));
  } catch (error) {
    showToast(error.message);
  }
}

async function clearOcrCache() {
  els.clearOcrCacheBtn.disabled = true;
  try {
    const data = await api("/ocr/cache/clear", {method: "POST"});
    const info = data.ocr_gpu_cache || {};
    const before = sumGpuMemory(info.memory_before, "reserved_mb");
    const after = sumGpuMemory(info.memory_after, "reserved_mb");
    const freed = Number(info.freed_reserved_mb ?? Math.max(0, before - after)).toFixed(0);
    const readersBefore = Number(info.readers_before || 0);
    const readersAfter = Number(info.readers_after || 0);
    if (!info.torch_loaded && readersBefore === 0) {
      showToast(t("toast.ocrCacheEmpty", {}, "OCR GPU cache уже пуст"));
    } else {
      showToast(t(
        "toast.ocrCacheCleared",
        {readersBefore, readersAfter, freed},
        "OCR GPU cache очищен: readers {readersBefore}->{readersAfter}, VRAM cache -{freed} MB",
      ));
    }
  } catch (error) {
    showToast(error.message);
  } finally {
    els.clearOcrCacheBtn.disabled = false;
  }
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function wire() {
  [
    "healthLine", "embeddingModelSelect", "languageSelect", "refreshBtn", "newIndexBtn",
    "deleteIndexBtn", "newIndexForm", "newIndexName", "indexList", "documentsTitle",
    "documentsHint", "uploadInput", "addPathBtn", "documentSearch", "documentStatusFilter",
    "stopJobsBtn", "jobsLine", "settingsBtn", "pathForm", "pathInput", "recursiveInput", "submitPathBtn",
    "selectVisibleInput", "selectUnindexedBtn", "selectedDocsLine",
    "bulkReindexBtn", "bulkForceReindexBtn", "bulkDeleteBtn", "clearSelectionBtn",
    "prevDocsBtn", "documentPageLine", "nextDocsBtn",
    "documentList", "detailsEmpty", "detailsContent", "detailName",
    "detailStatus", "detailChunks", "detailHash", "detailSource", "detailError",
    "reindexBtn", "deleteBtn", "settingsDialog", "ollamaUrlInput", "apiKeyInput",
    "requireApiKeyLocalhostInput",
    "embeddingProviderInput", "embeddingModelInput", "embeddingBaseUrlInput",
    "embeddingApiKeyInput", "embeddingBatchSizeInput", "embeddingCacheDtypeInput",
    "embeddingQueryPrefixInput", "embeddingDocumentPrefixInput",
    "defaultIndexSelect", "storageDirInput", "allowedRootsInput",
    "chunkSizeInput", "chunkOverlapInput", "indexTypeInput", "hnswThresholdInput",
    "hnswMInput", "hnswEfConstructionInput", "hnswEfSearchInput", "topKInput", "scoreInput",
    "ragEnabledInput", "includeSourcesInput", "deepAnalysisEnabledInput",
    "deepFinalAnswerInput", "deepForceAllInput", "retrievalTopKInput",
    "maxPromptChunksInput", "adaptiveScoreMarginInput", "minQueryTermHitsInput",
    "queryExpansionEnabledInput", "queryExpansionModelInput", "queryExpansionMaxVariantsInput",
    "queryExpansionMaxTokensInput",
    "rerankEnabledInput", "rerankModelInput", "rerankMinResultsInput", "rerankTopNInput",
    "maxContextCharsInput", "contextBatchCharsInput", "maxContextBatchesInput",
    "maxCompactSourcesInput", "contextTemplateInput",
    "chatAttachmentsEnabledInput", "chatAttachmentIndexPrefixInput", "chatAttachmentMaxFilesInput",
    "chatAttachmentMaxFileMbInput", "chatAttachmentTimeoutInput",
    "deepGenerationProviderInput", "deepGenerationBaseUrlInput", "deepGenerationApiKeyInput",
    "deepGenerationModelInput", "deepTopKInput", "deepMaxBatchesInput",
    "deepBatchCharsInput", "deepBatchTokensInput", "deepFinalTokensInput",
    "deepTimeoutInput", "deepTriggerPhrasesInput",
    "complianceEnabledInput", "complianceAutoInput", "complianceAllowUserIndexInput",
    "complianceIndexSelect", "complianceGenerationModelInput", "complianceMaxFilesInput",
    "complianceMaxFileMbInput", "complianceSectionCharsInput", "complianceMaxSectionsInput",
    "complianceTopKInput", "complianceTimeoutInput", "complianceTriggerPhrasesInput",
    "ocrInput", "ocrEngineInput", "ocrGpuInput", "ocrGpuDeviceInput", "ocrModelStorageInput",
    "pdfRenderScaleInput",
    "ocrLanguagesInput", "clearOcrCacheBtn", "saveSettingsBtn", "toast"
  ].forEach((id) => { els[id] = $(id); });

  els.languageSelect.addEventListener("change", async () => {
    await loadMessages(els.languageSelect.value);
    renderDefaultIndexOptions();
    renderDocuments();
    renderDetails();
    renderJobs();
    await Promise.allSettled([loadModels(), loadGenerationModels()]);
  });
  els.refreshBtn.addEventListener("click", refreshAll);
  els.newIndexBtn.addEventListener("click", () => els.newIndexForm.classList.toggle("is-hidden"));
  els.deleteIndexBtn.addEventListener("click", deleteActiveIndex);
  els.newIndexForm.addEventListener("submit", createIndex);
  els.indexList.addEventListener("click", async (event) => {
    const button = event.target.closest("[data-index-id]");
    if (!button) return;
    state.activeIndexId = button.dataset.indexId;
    state.activeDocumentId = "";
    state.documentOffset = 0;
    state.selectedDocumentIds.clear();
    renderIndexes();
    await loadDocuments();
    await loadJobs();
  });
  els.documentList.addEventListener("click", (event) => {
    const checkbox = event.target.closest(".doc-checkbox");
    if (checkbox) {
      const documentId = checkbox.dataset.documentId;
      if (checkbox.checked) {
        state.selectedDocumentIds.add(documentId);
      } else {
        state.selectedDocumentIds.delete(documentId);
      }
      renderBulkControls();
      return;
    }
    const button = event.target.closest("[data-document-id]");
    if (!button) return;
    state.activeDocumentId = button.dataset.documentId;
    renderDocuments();
    renderDetails();
  });
  els.documentSearch.addEventListener("input", scheduleDocumentReload);
  els.documentStatusFilter.addEventListener("change", scheduleDocumentReload);
  els.prevDocsBtn.addEventListener("click", () => changeDocumentPage(-1));
  els.nextDocsBtn.addEventListener("click", () => changeDocumentPage(1));
  els.uploadInput.addEventListener("change", uploadFiles);
  els.addPathBtn.addEventListener("click", () => els.pathForm.classList.toggle("is-hidden"));
  els.submitPathBtn.addEventListener("click", addPath);
  els.stopJobsBtn.addEventListener("click", stopIndexing);
  els.reindexBtn.addEventListener("click", reindexSelected);
  els.bulkReindexBtn.addEventListener("click", () => bulkReindexSelected(false));
  els.bulkForceReindexBtn.addEventListener("click", () => bulkReindexSelected(true));
  els.bulkDeleteBtn.addEventListener("click", bulkDeleteSelected);
  els.selectVisibleInput.addEventListener("change", toggleVisibleSelection);
  els.selectUnindexedBtn.addEventListener("click", selectUnindexedDocuments);
  els.clearSelectionBtn.addEventListener("click", clearDocumentSelection);
  els.deleteBtn.addEventListener("click", deleteSelected);
  els.settingsBtn.addEventListener("click", () => els.settingsDialog.showModal());
  els.saveSettingsBtn.addEventListener("click", saveSettings);
  els.clearOcrCacheBtn.addEventListener("click", clearOcrCache);
  els.embeddingProviderInput.addEventListener("change", loadModels);
  els.deepGenerationProviderInput.addEventListener("change", loadGenerationModels);
  els.deepGenerationBaseUrlInput.addEventListener("change", loadGenerationModels);
  els.embeddingModelSelect.addEventListener("change", () => {
    if (els.embeddingModelSelect.value) {
      els.embeddingModelInput.value = els.embeddingModelSelect.value;
    }
  });
  els.embeddingModelInput.addEventListener("change", () => {
    if (els.embeddingModelInput.value) {
      els.embeddingModelSelect.value = els.embeddingModelInput.value;
    }
  });
}

window.addEventListener("DOMContentLoaded", async () => {
  wire();
  await loadMessages();
  await refreshAll();
  window.setInterval(async () => {
    await loadJobs();
    const activeJobs = state.jobs.some((item) => ["queued", "running", "cancel_requested"].includes(item.status));
    if (activeJobs || Date.now() - state.lastDocumentRefresh > 30000) {
      await loadDocuments();
    }
  }, 5000);
});
