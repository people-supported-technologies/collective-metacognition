// D3 force-directed viewer for attributed causal maps.
// Fixed node universe; filters resize nodes/edges. Polarity colouring and
// group comparison highlight where different demographics' maps diverge.

const NODE_TYPE_LABELS = {
    actor: "Actor",
    technology: "Technology",
    policy: "Policy",
    event: "Event",
    process: "Process",
    outcome: "Outcome",
    value: "Value",
    belief_proposition: "Belief proposition",
    topic: "Topic",
};

const NODE_TYPE_COLORS = {
    actor: "#f97316",
    technology: "#06b6d4",
    policy: "#8b5cf6",
    event: "#ef4444",
    process: "#22c55e",
    outcome: "#eab308",
    value: "#ec4899",
    belief_proposition: "#14b8a6",
    topic: "#6366f1",
};

const POL_SUPPORT = "#16a34a", POL_NEUTRAL = "#9ca3af", POL_OPPOSE = "#dc2626";
const GROUP_A_COLOR = "#2563eb", GROUP_B_COLOR = "#dc2626", BOTH_COLOR = "#9ca3af";

// Interpolate a polarity sign in [-1, 1] to a colour (red -> gray -> green).
function polarityColor(mean) {
    if (mean == null || isNaN(mean)) return POL_NEUTRAL;
    if (mean >= 0) return d3.interpolateRgb(POL_NEUTRAL, POL_SUPPORT)(Math.min(1, mean));
    return d3.interpolateRgb(POL_NEUTRAL, POL_OPPOSE)(Math.min(1, -mean));
}

(async function() {
    const data = await d3.json("graph_data.json");
    if (!data) { console.error("Failed to load graph_data.json"); return; }

    const {
        nodes,
        raw_edges,
        raw_stances = [],
        metadata,
        participant_demographics = {},
    } = data;

    // ---- Multi-dimension demographic segments (intersection filter) ----
    const JUNK_VALUES = new Set(["CONSENT_REVOKED", "DATA_EXPIRED", "", "Unknown", "nan"]);

    function ageGroup(age) {
        const n = parseInt(age, 10);
        if (isNaN(n)) return undefined;
        if (n < 30) return "18-29";
        if (n < 45) return "30-44";
        if (n < 60) return "45-59";
        return "60+";
    }

    // Build an augmented per-participant demographic record with derived age_group
    // and junk values stripped.
    const pdemo = {};
    for (const [pid, raw] of Object.entries(participant_demographics)) {
        const rec = {};
        for (const key of ["political_affiliation", "sex", "ethnicity"]) {
            const v = raw[key];
            if (v && !JUNK_VALUES.has(v)) rec[key] = v;
        }
        const ag = ageGroup(raw.age);
        if (ag) rec.age_group = ag;
        pdemo[pid] = rec;
    }

    const DIM_CONFIG = [
        { key: "political_affiliation", label: "Political affiliation" },
        { key: "age_group", label: "Age group", order: ["18-29", "30-44", "45-59", "60+"] },
        { key: "sex", label: "Sex" },
        { key: "ethnicity", label: "Ethnicity" },
    ];

    // Distinct values per dimension (with participant counts) over participants who speak.
    const speakerSet = new Set(metadata.speakers || []);
    const DIMS = DIM_CONFIG.map(cfg => {
        const counts = new Map();
        for (const pid of speakerSet) {
            const v = pdemo[pid]?.[cfg.key];
            if (v) counts.set(v, (counts.get(v) || 0) + 1);
        }
        let values = [...counts.entries()];
        if (cfg.order) {
            values.sort((a, b) => cfg.order.indexOf(a[0]) - cfg.order.indexOf(b[0]));
        } else {
            values.sort((a, b) => b[1] - a[1]);
        }
        return { key: cfg.key, label: cfg.label, values };
    }).filter(d => d.values.length > 0);

    function buildDimensionSelects(containerId, prefix) {
        const container = document.getElementById(containerId);
        container.innerHTML = "";
        for (const dim of DIMS) {
            const label = document.createElement("label");
            label.textContent = dim.label;
            const sel = document.createElement("select");
            sel.id = `${prefix}-${dim.key}`;
            sel.dataset.dimKey = dim.key;
            sel.dataset.segment = prefix;
            const anyOpt = document.createElement("option");
            anyOpt.value = "any"; anyOpt.textContent = "Any";
            sel.appendChild(anyOpt);
            for (const [val, count] of dim.values) {
                const opt = document.createElement("option");
                opt.value = val; opt.textContent = `${val} (${count})`;
                sel.appendChild(opt);
            }
            sel.addEventListener("change", update);
            container.appendChild(label);
            container.appendChild(sel);
        }
    }
    buildDimensionSelects("segment-a-dims", "dimA");
    buildDimensionSelects("segment-b-dims", "dimB");

    function readSegment(prefix) {
        const sel = {};
        for (const dim of DIMS) {
            const el = document.getElementById(`${prefix}-${dim.key}`);
            if (el && el.value !== "any") sel[dim.key] = el.value;
        }
        return sel;
    }

    function matchesSegment(pid, segment) {
        const rec = pdemo[pid];
        for (const [key, val] of Object.entries(segment)) {
            if (!rec || rec[key] !== val) return false;
        }
        return true;
    }

    function describeSegment(segment) {
        const parts = DIMS.map(d => segment[d.key]).filter(Boolean);
        return parts.length ? parts.join(", ") : "All participants";
    }

    populateSelect("filter-round", metadata.rounds.map(r => ({ value: r, label: `Round ${r}` })));
    populateSelect("filter-table", metadata.tables.map(t => ({ value: t, label: t.slice(0, 8) })));
    populateSelect("filter-speaker", metadata.speakers.map(s => ({ value: s, label: s.slice(0, 12) })));

    const typeCounts = new Map();
    for (const n of nodes) typeCounts.set(n.type, (typeCounts.get(n.type) || 0) + 1);
    const entityTypeOptions = [
        { value: "all", label: "All types", count: nodes.length },
        ...[...typeCounts.entries()].sort((a, b) => b[1] - a[1])
            .map(([type, count]) => ({ value: type, label: NODE_TYPE_LABELS[type] || type, count })),
    ];
    populateSelect("filter-entity-type", entityTypeOptions.map(opt => ({
        value: opt.value, label: opt.count != null ? `${opt.label} (${opt.count})` : opt.label,
    })));
    buildTypeLegend(typeCounts);

    document.getElementById("stat-nodes").textContent = nodes.length;
    document.getElementById("stat-edges").textContent = raw_edges.length;
    document.getElementById("stat-stances").textContent = raw_stances.length;

    const container = document.getElementById("graph-container");
    const width = container.clientWidth;
    const height = container.clientHeight;

    const svg = d3.select("#graph-container").append("svg").attr("width", width).attr("height", height);
    svg.append("defs").append("marker")
        .attr("id", "arrowhead").attr("viewBox", "0 -5 10 10")
        .attr("refX", 20).attr("refY", 0).attr("markerWidth", 6).attr("markerHeight", 6)
        .attr("orient", "auto").append("path").attr("d", "M0,-5L10,0L0,5").attr("class", "arrow-marker");

    const g = svg.append("g");
    svg.call(d3.zoom().scaleExtent([0.1, 5]).on("zoom", (e) => g.attr("transform", e.transform)));

    const nodeMap = new Map(nodes.map(n => [n.node_id, { ...n, x: width / 2, y: height / 2 }]));
    const nodeArray = Array.from(nodeMap.values());
    let curDescA = "A", curDescB = "B";  // current segment descriptions (for labels)

    const simulation = d3.forceSimulation(nodeArray)
        .force("link", d3.forceLink().id(d => d.node_id).distance(80))
        .force("charge", d3.forceManyBody().strength(-50))
        .force("center", d3.forceCenter(width / 2, height / 2))
        .force("collision", d3.forceCollide().radius(20))
        .force("x", d3.forceX(width / 2).strength(0.05))
        .force("y", d3.forceY(height / 2).strength(0.05));

    // After the initial layout converges, freeze all node positions so that
    // switching demographic filters only changes visual encoding (color, size,
    // link presence) without moving nodes — essential for comparing conditions.
    let layoutFrozen = false;
    simulation.on("end", () => {
        if (!layoutFrozen) {
            layoutFrozen = true;
            for (const n of nodeArray) { n.fx = n.x; n.fy = n.y; }
        }
    });

    const linkGroup = g.append("g").attr("class", "links");
    const labelGroup = g.append("g").attr("class", "edge-labels");
    const nodeGroup = g.append("g").attr("class", "nodes");
    const textGroup = g.append("g").attr("class", "node-texts");

    const tooltip = document.getElementById("tooltip");
    const modalOverlay = document.getElementById("modal-overlay");
    const modal = document.getElementById("modal");
    modalOverlay.addEventListener("click", (e) => { if (e.target === modalOverlay) closeModal(); });
    document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeModal(); });

    function showTooltip(event, html) {
        tooltip.innerHTML = html; tooltip.style.display = "block";
        tooltip.style.left = (event.pageX + 12) + "px"; tooltip.style.top = (event.pageY - 10) + "px";
    }
    function hideTooltip() { tooltip.style.display = "none"; }
    function moveTooltip(event) { tooltip.style.left = (event.pageX + 12) + "px"; tooltip.style.top = (event.pageY - 10) + "px"; }
    function openModal(html) { modal.innerHTML = html; modalOverlay.classList.add("active"); }
    function closeModal() { modalOverlay.classList.remove("active"); }

    function affiliationOf(pid) { return participant_demographics[pid]?.political_affiliation || "Unknown"; }
    function getNodeType(nodeId) { return nodeMap.get(nodeId)?.type; }

    // --- shared non-group filter predicates ---
    function passesCommon(e) {
        const entityType = document.getElementById("filter-entity-type").value;
        const round = document.getElementById("filter-round").value;
        const table = document.getElementById("filter-table").value;
        const speaker = document.getElementById("filter-speaker").value;
        const minConf = parseFloat(document.getElementById("filter-confidence").value);
        const stance = document.getElementById("filter-stance").value;
        const showInferred = document.getElementById("toggle-inferred").checked;

        if (entityType !== "all") {
            const srcType = getNodeType(e.source_node_id);
            const tgtType = getNodeType(e.target_node_id);
            if (srcType !== entityType && tgtType !== entityType) return false;
        }
        if (round !== "all" && e.round_id !== round) return false;
        if (table !== "all" && e.table_id !== table) return false;
        if (speaker !== "all" && e.participant_id !== speaker) return false;
        if (e.confidence < minConf) return false;
        if (stance !== "all" && e.stance !== stance) return false;
        if (!showInferred && e.explicitness === "inferred") return false;
        return true;
    }

    function stancePassesCommon(s) {
        const entityType = document.getElementById("filter-entity-type").value;
        const round = document.getElementById("filter-round").value;
        const table = document.getElementById("filter-table").value;
        const speaker = document.getElementById("filter-speaker").value;
        const minConf = parseFloat(document.getElementById("filter-confidence").value);
        const showInferred = document.getElementById("toggle-inferred").checked;
        if (entityType !== "all" && getNodeType(s.concept_node_id) !== entityType) return false;
        if (round !== "all" && s.round_id !== round) return false;
        if (table !== "all" && s.table_id !== table) return false;
        if (speaker !== "all" && s.participant_id !== speaker) return false;
        if (s.confidence < minConf) return false;
        if (!showInferred && s.explicitness === "inferred") return false;
        return true;
    }

    function getEdges(segment) {
        return raw_edges.filter(e => passesCommon(e) && matchesSegment(e.participant_id, segment));
    }
    function getStances(segment) {
        return raw_stances.filter(s => stancePassesCommon(s) && matchesSegment(s.participant_id, segment));
    }

    // Per-node mean polarity from edges (both endpoints) + stances.
    function nodePolarity(edges, stances) {
        const acc = new Map();
        const add = (nid, sign) => {
            if (!nid) return;
            const a = acc.get(nid) || { sum: 0, n: 0 };
            a.sum += sign; a.n += 1; acc.set(nid, a);
        };
        for (const e of edges) { add(e.source_node_id, e.polarity_sign); add(e.target_node_id, e.polarity_sign); }
        for (const s of stances) add(s.concept_node_id, s.polarity_sign);
        const out = new Map();
        for (const [nid, a] of acc) out.set(nid, a.n ? a.sum / a.n : 0);
        return out;
    }

    // Degree centrality = number of distinct neighbouring concepts, from a set of
    // aggregated links (source/target may be ids or node objects).
    function degreeFromLinks(linkList) {
        const neighbors = new Map();
        const add = (a, b) => {
            if (!neighbors.has(a)) neighbors.set(a, new Set());
            neighbors.get(a).add(b);
        };
        for (const l of linkList) {
            const s = l.source.node_id || l.source;
            const t = l.target.node_id || l.target;
            if (!s || !t) continue;
            add(s, t); add(t, s);
        }
        const deg = new Map();
        for (const [id, set] of neighbors) deg.set(id, set.size);
        return deg;
    }

    const TOP_CENTRAL = 15;

    function aggregateEdges(filtered) {
        const groups = new Map();
        for (const e of filtered) {
            const key = `${e.source_node_id}|${e.target_node_id}|${e.relation}`;
            if (!groups.has(key)) groups.set(key, []);
            groups.get(key).push(e);
        }
        return Array.from(groups.entries()).map(([key, edges]) => {
            const [source, target, relation] = key.split("|");
            const speakers = new Set(edges.map(e => e.participant_id));
            const confs = edges.map(e => e.confidence);
            const signs = edges.map(e => e.polarity_sign);
            const hasInferred = edges.some(e => e.explicitness === "inferred" || e.explicitness === "near_explicit");
            return {
                source, target, relation,
                count: edges.length,
                unique_speakers: speakers.size,
                mean_confidence: confs.reduce((a, b) => a + b, 0) / confs.length,
                mean_polarity: signs.reduce((a, b) => a + b, 0) / signs.length,
                has_inferred: hasInferred,
                evidence: edges.map(e => ({
                    text: e.evidence_text, speaker: e.speaker, stance: e.stance,
                    polarity: e.polarity, affiliation: affiliationOf(e.participant_id),
                })),
            };
        });
    }

    function update() {
        const segmentA = readSegment("dimA");
        const segmentB = readSegment("dimB");
        const colorMode = document.getElementById("color-mode").value;
        const hideIsolated = document.getElementById("toggle-isolated").checked;
        const showLabels = document.getElementById("toggle-labels").checked;
        const highlightCentral = document.getElementById("toggle-central").checked;
        const comparing = document.getElementById("toggle-compare").checked;
        document.getElementById("central-hint").style.display = highlightCentral ? "" : "none";
        document.getElementById("segment-b-wrap").style.display = comparing ? "" : "none";

        const descA = describeSegment(segmentA);
        const descB = describeSegment(segmentB);
        curDescA = descA; curDescB = descB;
        document.getElementById("stat-segment").textContent = descA;

        // Toggle legends
        document.getElementById("type-legend").style.display = (colorMode === "type" && !comparing) ? "" : "none";
        document.getElementById("polarity-legend").style.display = (colorMode === "polarity" && !comparing) ? "" : "none";
        document.getElementById("compare-legend").style.display = comparing ? "" : "none";
        document.getElementById("legend-title").textContent = comparing ? "Comparison" : "Node colours";
        document.getElementById("stat-compare-wrap").style.display = comparing ? "" : "none";
        if (comparing) {
            document.getElementById("legend-only-a").textContent = `Only A (${descA})`;
            document.getElementById("legend-only-b").textContent = `Only B (${descB})`;
        }

        let links, nodeWeights, nodePol, divergence, edgePresence;
        let degreeA = null, degreeB = null;

        if (!comparing) {
            const filtered = getEdges(segmentA);
            const filteredStances = getStances(segmentA);
            const agg = aggregateEdges(filtered);
            document.getElementById("stat-visible").textContent = filtered.length;

            nodeWeights = new Map(nodes.map(n => [n.node_id, 0]));
            for (const e of filtered) {
                nodeWeights.set(e.source_node_id, (nodeWeights.get(e.source_node_id) || 0) + 1);
                nodeWeights.set(e.target_node_id, (nodeWeights.get(e.target_node_id) || 0) + 1);
            }
            for (const s of filteredStances) nodeWeights.set(s.concept_node_id, (nodeWeights.get(s.concept_node_id) || 0) + 1);

            nodePol = nodePolarity(filtered, filteredStances);
            links = agg.map(e => ({ ...e, source: e.source, target: e.target, presence: "both" }));
        } else {
            // Comparison: union of A and B edges, tagged by segment presence.
            const edgesA = getEdges(segmentA), edgesB = getEdges(segmentB);
            const stancesA = getStances(segmentA), stancesB = getStances(segmentB);

            const aggA = aggregateEdges(edgesA), aggB = aggregateEdges(edgesB);
            degreeA = degreeFromLinks(aggA);
            degreeB = degreeFromLinks(aggB);
            const keyOf = e => `${e.source}|${e.target}|${e.relation}`;
            const mapA = new Map(aggA.map(e => [keyOf(e), e]));
            const mapB = new Map(aggB.map(e => [keyOf(e), e]));
            const allKeys = new Set([...mapA.keys(), ...mapB.keys()]);

            links = [];
            for (const key of allKeys) {
                const a = mapA.get(key), b = mapB.get(key);
                const base = a || b;
                const presence = a && b ? "both" : (a ? "a" : "b");
                links.push({
                    ...base,
                    source: base.source, target: base.target, relation: base.relation,
                    count: (a?.count || 0) + (b?.count || 0),
                    unique_speakers: (a?.unique_speakers || 0) + (b?.unique_speakers || 0),
                    mean_confidence: base.mean_confidence,
                    mean_polarity: base.mean_polarity,
                    has_inferred: base.has_inferred,
                    presence,
                    evidence: [...(a?.evidence || []), ...(b?.evidence || [])],
                });
            }
            document.getElementById("stat-visible").textContent = edgesA.length + edgesB.length;

            const polA = nodePolarity(edgesA, stancesA);
            const polB = nodePolarity(edgesB, stancesB);
            nodePol = polA; // node fill uses group A position
            nodeWeights = new Map(nodes.map(n => [n.node_id, 0]));
            const bump = (nid) => nid && nodeWeights.set(nid, (nodeWeights.get(nid) || 0) + 1);
            for (const e of edgesA.concat(edgesB)) { bump(e.source_node_id); bump(e.target_node_id); }
            for (const s of stancesA.concat(stancesB)) bump(s.concept_node_id);

            divergence = new Map();
            const allNodeIds = new Set([...polA.keys(), ...polB.keys()]);
            let divCount = 0;
            for (const nid of allNodeIds) {
                const d = Math.abs((polA.get(nid) || 0) - (polB.get(nid) || 0));
                divergence.set(nid, d);
                if (d >= 0.5) divCount += 1;
            }
            document.getElementById("stat-diverge").textContent = divCount;
        }

        // Centrality: per-segment degree, or differential degree in compare mode.
        const centralityScore = new Map();
        if (comparing) {
            const maxA = Math.max(1, ...degreeA.values());
            const maxB = Math.max(1, ...degreeB.values());
            const ids = new Set([...degreeA.keys(), ...degreeB.keys()]);
            for (const id of ids) {
                const normA = (degreeA.get(id) || 0) / maxA;
                const normB = (degreeB.get(id) || 0) / maxB;
                centralityScore.set(id, Math.abs(normA - normB));
            }
        } else {
            for (const [id, deg] of degreeFromLinks(links)) centralityScore.set(id, deg);
        }
        const topCentral = new Set();
        if (highlightCentral) {
            const ranked = [...centralityScore.entries()]
                .filter(([id, s]) => s > 0 && (nodeWeights.get(id) || 0) > 0)
                .sort((a, b) => b[1] - a[1])
                .slice(0, TOP_CENTRAL)
                .map(([id]) => id);
            ranked.forEach(id => topCentral.add(id));
        }

        simulation.force("link").links(links);

        // Links
        const linkSel = linkGroup.selectAll("line").data(links, d => `${d.source.node_id || d.source}|${d.target.node_id || d.target}|${d.relation}`);
        linkSel.exit().remove();
        const linkEnter = linkSel.enter().append("line")
            .attr("class", "edge-line").attr("cursor", "pointer").attr("marker-end", "url(#arrowhead)")
            .on("mouseenter", (event, d) => showTooltip(event, `
                <div class="tt-rel">${d.relation}</div>
                <div class="tt-label">${getNodeLabel(d.source)} &rarr; ${getNodeLabel(d.target)}</div>
                <div class="tt-type">${d.count} mention(s) | ${d.unique_speakers} speaker(s) | polarity: ${fmtPol(d.mean_polarity)}${comparing ? " | " + presenceLabel(d.presence) : ""}</div>`))
            .on("mousemove", moveTooltip).on("mouseleave", hideTooltip)
            .on("click", (event, d) => { hideTooltip(); openEdgeModal(d, comparing); });
        linkEnter.merge(linkSel)
            .attr("stroke", d => comparing ? presenceColor(d.presence) : "#64748b")
            .attr("stroke-width", d => comparing && d.presence !== "both" ? Math.max(2, Math.min(7, d.count)) : Math.max(1, Math.min(6, d.unique_speakers * 1.5)))
            .attr("stroke-opacity", d => 0.35 + d.mean_confidence * 0.65)
            .attr("stroke-dasharray", d => d.has_inferred ? "4,3" : null);

        const edgeLabelSel = labelGroup.selectAll("text").data(showLabels ? links : [], d => `${d.source.node_id || d.source}|${d.target.node_id || d.target}|${d.relation}`);
        edgeLabelSel.exit().remove();
        edgeLabelSel.enter().append("text").attr("class", "edge-label").merge(edgeLabelSel).text(d => d.relation);

        // Nodes
        const maxWeight = Math.max(1, ...nodeWeights.values());
        const nodeData = nodeArray.filter(n => !hideIsolated || nodeWeights.get(n.node_id) > 0);

        const nodeSel = nodeGroup.selectAll("circle").data(nodeData, d => d.node_id);
        nodeSel.exit().remove();
        const nodeEnter = nodeSel.enter().append("circle").attr("cursor", "pointer")
            .call(d3.drag().on("start", dragstarted).on("drag", dragged).on("end", dragended))
            .on("mouseenter", (event, d) => {
                const w = nodeWeights.get(d.node_id) || 0;
                const pol = nodePol.get(d.node_id);
                const div = divergence ? divergence.get(d.node_id) : null;
                const cen = centralityScore.get(d.node_id);
                showTooltip(event, `
                    <div class="tt-label">${d.label}</div>
                    <div class="tt-type">${d.type} | weight: ${w} | polarity: ${fmtPol(pol)}</div>
                    ${div != null ? `<div class="tt-type">divergence: ${div.toFixed(2)}</div>` : ""}
                    ${cen != null ? `<div class="tt-type">${comparing ? "diff. centrality" : "centrality"}: ${cen.toFixed(2)}</div>` : ""}`);
            })
            .on("mousemove", moveTooltip).on("mouseleave", hideTooltip)
            .on("click", (event, d) => { hideTooltip(); openNodeModal(d, nodeWeights.get(d.node_id), nodePol.get(d.node_id), links); });
        nodeEnter.merge(nodeSel)
            .attr("r", d => {
                const w = nodeWeights.get(d.node_id) || 0;
                const base = w === 0 ? 4 : 6 + (w / maxWeight) * 20;
                return topCentral.has(d.node_id) ? base + 6 : base;
            })
            .attr("fill", d => {
                if (comparing) return polarityColor(nodePol.get(d.node_id));
                return colorMode === "polarity" ? polarityColor(nodePol.get(d.node_id)) : nodeColor(d.type);
            })
            .attr("stroke", d => {
                if (topCentral.has(d.node_id)) return "#f59e0b";
                if (comparing) {
                    const div = divergence.get(d.node_id) || 0;
                    return div >= 0.5 ? "#7c3aed" : "#ffffff";
                }
                return "#ffffff";
            })
            .attr("stroke-width", d => {
                if (topCentral.has(d.node_id)) return 4;
                if (comparing) { const div = divergence.get(d.node_id) || 0; return div >= 0.5 ? 3 : 1.5; }
                return 1.5;
            })
            .attr("opacity", d => {
                const w = nodeWeights.get(d.node_id) || 0;
                if (highlightCentral) return topCentral.has(d.node_id) ? 1 : (w > 0 ? 0.15 : 0.05);
                return w > 0 ? 1 : 0.25;
            });

        const labelData = highlightCentral
            ? nodeData.filter(d => topCentral.has(d.node_id))
            : nodeData.filter(d => nodeWeights.get(d.node_id) > 0);
        const textSel = textGroup.selectAll("text").data(labelData, d => d.node_id);
        textSel.exit().remove();
        textSel.enter().append("text").attr("class", "node-label").attr("dy", -14)
            .merge(textSel).text(d => d.label.length > 22 ? d.label.slice(0, 20) + "..." : d.label);

        simulation.nodes(nodeData);
        if (!layoutFrozen) {
            simulation.alpha(0.3).restart();
        } else {
            // Positions frozen: just redraw at current coords without physics.
            simulation.alpha(0).restart();
            tickPositions();
        }
    }

    function tickPositions() {
        linkGroup.selectAll("line").attr("x1", d => d.source.x).attr("y1", d => d.source.y).attr("x2", d => d.target.x).attr("y2", d => d.target.y);
        labelGroup.selectAll("text").attr("x", d => (d.source.x + d.target.x) / 2).attr("y", d => (d.source.y + d.target.y) / 2);
        nodeGroup.selectAll("circle").attr("cx", d => d.x).attr("cy", d => d.y);
        textGroup.selectAll("text").attr("x", d => d.x).attr("y", d => d.y);
    }

    simulation.on("tick", tickPositions);

    function dragstarted(event, d) { if (!event.active) simulation.alphaTarget(0.3).restart(); d.fx = d.x; d.fy = d.y; }
    function dragged(event, d) { d.fx = event.x; d.fy = event.y; }
    function dragended(event, d) {
        if (!event.active) simulation.alphaTarget(0);
        if (layoutFrozen) { d.fx = event.x; d.fy = event.y; }
        else { d.fx = null; d.fy = null; }
    }

    function nodeColor(type) { return NODE_TYPE_COLORS[type] || "#8899a6"; }
    function presenceColor(p) { return p === "a" ? GROUP_A_COLOR : p === "b" ? GROUP_B_COLOR : BOTH_COLOR; }
    function presenceLabel(p) {
        if (p === "a") return `only A (${curDescA})`;
        if (p === "b") return `only B (${curDescB})`;
        return "both segments";
    }
    function fmtPol(v) {
        if (v == null || isNaN(v)) return "n/a";
        const tag = v > 0.15 ? "supports" : v < -0.15 ? "opposes" : "neutral";
        return `${v.toFixed(2)} (${tag})`;
    }

    function buildTypeLegend(typeCounts) {
        const legend = document.getElementById("type-legend");
        const sorted = [...typeCounts.entries()].sort((a, b) => b[1] - a[1]);
        legend.innerHTML = sorted.map(([type, count]) => `
            <div class="legend-item">
                <span class="legend-swatch" style="background:${nodeColor(type)}"></span>
                <span>${NODE_TYPE_LABELS[type] || type}</span>
                <span class="legend-count">${count}</span>
            </div>`).join("");
    }

    function openNodeModal(node, weight, pol, links) {
        const connected = links.filter(e => (e.source.node_id || e.source) === node.node_id || (e.target.node_id || e.target) === node.node_id);
        const outgoing = connected.filter(e => (e.source.node_id || e.source) === node.node_id);
        const incoming = connected.filter(e => (e.target.node_id || e.target) === node.node_id);

        let html = "";
        if (outgoing.length) {
            html += `<div class="modal-section"><h4>Outgoing (${outgoing.length})</h4>`;
            outgoing.forEach(e => html += `<div class="modal-evidence"><strong>${e.relation}</strong> &rarr; ${getNodeLabel(e.target)}<div class="ev-meta">${e.count} mention(s) | polarity ${fmtPol(e.mean_polarity)}</div></div>`);
            html += `</div>`;
        }
        if (incoming.length) {
            html += `<div class="modal-section"><h4>Incoming (${incoming.length})</h4>`;
            incoming.forEach(e => html += `<div class="modal-evidence">${getNodeLabel(e.source)} &rarr; <strong>${e.relation}</strong><div class="ev-meta">${e.count} mention(s) | polarity ${fmtPol(e.mean_polarity)}</div></div>`);
            html += `</div>`;
        }
        openModal(`
            <button class="modal-close" onclick="document.getElementById('modal-overlay').classList.remove('active')">&times;</button>
            <h2>${node.label}</h2>
            <div class="modal-subtitle">${node.type}</div>
            <div class="modal-stats">
                <div class="stat-card"><div class="stat-val">${weight || 0}</div><div class="stat-lbl">Mentions</div></div>
                <div class="stat-card"><div class="stat-val">${fmtPol(pol).split(" ")[0]}</div><div class="stat-lbl">Polarity</div></div>
                <div class="stat-card"><div class="stat-val">${outgoing.length + incoming.length}</div><div class="stat-lbl">Links</div></div>
            </div>
            ${node.aliases && node.aliases.length > 1 ? `<div class="modal-section" style="margin-top:16px;"><h4>Aliases</h4><p style="font-size:13px;">${node.aliases.join(", ")}</p></div>` : ""}
            ${html || `<div class="modal-section"><p style="color:#9ca3af;">No connections under current filter.</p></div>`}`);
    }

    function openEdgeModal(edge, comparing) {
        const evidenceHtml = edge.evidence.map(e => `
            <div class="modal-evidence">"${e.text}"
                <div class="ev-meta">${e.speaker} | ${e.affiliation || "Unknown"} | stance: ${e.stance} | polarity: ${e.polarity}</div>
            </div>`).join("");
        openModal(`
            <button class="modal-close" onclick="document.getElementById('modal-overlay').classList.remove('active')">&times;</button>
            <h2>${getNodeLabel(edge.source)} &rarr; ${getNodeLabel(edge.target)}</h2>
            <div class="modal-subtitle">${edge.relation}</div>
            <div class="modal-stats">
                <div class="stat-card"><div class="stat-val">${edge.count}</div><div class="stat-lbl">Mentions</div></div>
                <div class="stat-card"><div class="stat-val">${edge.unique_speakers}</div><div class="stat-lbl">Speakers</div></div>
                <div class="stat-card"><div class="stat-val">${fmtPol(edge.mean_polarity).split(" ")[0]}</div><div class="stat-lbl">Polarity</div></div>
            </div>
            ${comparing ? `<div class="modal-section" style="margin-top:16px;"><span class="modal-badge badge-stance">${presenceLabel(edge.presence)}</span></div>` : ""}
            <div class="modal-section"><h4>Evidence (${edge.evidence.length})</h4>${evidenceHtml}</div>`);
    }

    function getNodeLabel(idOrObj) {
        if (typeof idOrObj === "object") return idOrObj.label || idOrObj.node_id;
        const n = nodeMap.get(idOrObj);
        return n ? n.label : idOrObj;
    }

    function populateSelect(id, options) {
        const sel = document.getElementById(id);
        for (const opt of options) {
            const el = document.createElement("option");
            el.value = opt.value; el.textContent = opt.label; sel.appendChild(el);
        }
    }

    ["color-mode", "filter-entity-type", "filter-round", "filter-table", "filter-speaker", "filter-stance", "toggle-compare"].forEach(id => {
        document.getElementById(id).addEventListener("change", update);
    });
    document.getElementById("filter-confidence").addEventListener("input", (e) => {
        document.getElementById("conf-val").textContent = e.target.value; update();
    });
    document.getElementById("filter-repulsion").addEventListener("input", (e) => {
        const val = parseFloat(e.target.value);
        document.getElementById("repulsion-val").textContent = val;
        simulation.force("charge", d3.forceManyBody().strength(-val));
        layoutFrozen = false;
        for (const n of nodeArray) { n.fx = null; n.fy = null; }
        simulation.alpha(0.5).restart();
    });
    ["toggle-inferred", "toggle-labels", "toggle-isolated", "toggle-central"].forEach(id => {
        document.getElementById(id).addEventListener("change", update);
    });

    update();
})();
