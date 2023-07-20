<script>
	export let data;
</script>

<ul>
	{#each data.posts as post}
		<li>
			<span class="date">{post.meta.date}</span>
			<a href={post.path}>
				{post.meta.title}
			</a>
		</li>
	{/each}
</ul>

<style>
	ul {
		list-style-type: none;
	}
	li {
		padding: 12px;
	}
	.date {
		color: #888;
		font-size: 0.75em;
		padding: 4px;
	}
</style>
